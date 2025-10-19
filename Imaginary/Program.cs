/*
DETAILED PSEUDOCODE PLAN:
- At program start:
  - Parse command line arguments for:
    - --input-directory (required)
    - --output-directory (optional)
    - --try-model (optional)
    - --threshold (optional; double, invariant culture)
    - --delete-from-source (optional; bool)
    - --help / -h (prints usage and exits)
  - Validate input directory exists; print usage and exit on error.
  - Validate model path exists; print error and exit on error.
  - Derive duplicateFolder (output directory) defaulting to Path.GetTempPath()/Duplicates.
  - Create duplicateFolder directory.
- Create an ONNX InferenceSession for the model.
- Find image files recursively under input-directory and generate embeddings concurrently:
  - Limit concurrent session.Run calls with a SemaphoreSlim.
  - Store embeddings in a ConcurrentDictionary keyed by full image path.
  - Log successes and failures.
- Compare all embeddings in parallel:
  - Use a ConcurrentDictionary to mark files already moved.
  - If cosine similarity > threshold, try to atomically mark and move the duplicate file to duplicateFolder.
  - Optionally delete from source if flag set.
  - Log moves and failures.
- After processing:
  - Print "Processing complete!".
  - Attempt to open the duplicateFolder in the user's file explorer:
    - On Windows: use ProcessStartInfo with UseShellExecute = true and FileName = folder path.
    - On macOS: run "open <folder>".
    - On Linux: run "xdg-open <folder>".
    - Catch and log exceptions if opening fails.
  - Wait for user input before exiting (Console.ReadLine()).
- Keep all existing embedding, normalization, similarity, and image helpers unchanged.
*/

using System;
using System.Linq;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Imaginary
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // If user asked for help
            if (args.Any(a => a == "--help" || a == "-h"))
            {
                PrintUsage();
                return;
            }

            // Helper to get argument values
            static string? GetArgValue(string[] argv, string name)
            {
                string prefix = "--" + name;
                for (int i = 0; i < argv.Length; i++)
                {
                    var a = argv[i];
                    if (a.StartsWith(prefix + "=", StringComparison.Ordinal))
                    {
                        return a.Substring(prefix.Length + 1);
                    }
                    if (string.Equals(a, prefix, StringComparison.Ordinal) && i + 1 < argv.Length)
                    {
                        return argv[i + 1];
                    }
                }
                return null;
            }

            var argDeleteFromSource = GetArgValue(args, "delete-from-source");
            var argSimilarityThreshold = GetArgValue(args, "threshold");
            var argModelPath = GetArgValue(args, "try-model");
            var argInputDirectory = GetArgValue(args, "input-directory");
            var argOutputDirectory = GetArgValue(args, "output-directory");

            double SimilarityThreshold = 0.95;
            if (!string.IsNullOrEmpty(argSimilarityThreshold) &&
                double.TryParse(argSimilarityThreshold, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var parsedThreshold))
            {
                SimilarityThreshold = parsedThreshold;
            }

            bool DeleteFromSource = false;
            if (!string.IsNullOrEmpty(argDeleteFromSource) &&
                bool.TryParse(argDeleteFromSource, out var parsedBool))
            {
                DeleteFromSource = parsedBool;
            }

            if (string.IsNullOrEmpty(argInputDirectory) || !Directory.Exists(argInputDirectory))
            {
                Console.WriteLine("Please provide a valid input directory using --input-directory <path>.");
                PrintUsage();
                return;
            }

            string modelPath = argModelPath ?? Path.Combine("Models", "resnet50-v2-7.onnx");
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Model file not found: {modelPath}");
                Console.WriteLine("Provide a model using --try-model <path> or place the model at the default Models\\resnet50-v2-7.onnx");
                return;
            }

            string inputFolder = argInputDirectory!;
            string duplicateFolder = argOutputDirectory ?? Path.Combine(Path.GetTempPath(), "Duplicates");

            Directory.CreateDirectory(duplicateFolder);

            var session = new InferenceSession(modelPath);
            var embeddings = new ConcurrentDictionary<string, float[]>();

            Console.WriteLine("Generating embeddings...");

            // Changed to traverse nested/child folders and filter to image files
            var files = Directory.GetFiles(inputFolder, "*.*", SearchOption.AllDirectories).Where(f => IsImageFile(f)).ToArray();

            var sessionLock = new SemaphoreSlim(Environment.ProcessorCount); // limit concurrent Run calls

            await Parallel.ForEachAsync(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, async (imagePath, ct) =>
            {
                try
                {
                    var emb = await GetImageEmbeddingAsync(session, imagePath, sessionLock);
                    embeddings[imagePath] = emb;
                    Console.WriteLine($"Embedded: {imagePath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed embedding {imagePath}: {ex.Message}");
                }
            });

            Console.WriteLine("Comparing images...");
            var kvList = embeddings.ToList();
            var moved = new ConcurrentDictionary<string, bool>();

            Parallel.For(0, kvList.Count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, i =>
            {
                var img1 = kvList[i].Key;
                if (moved.ContainsKey(img1)) return;

                for (int j = i + 1; j < kvList.Count; j++)
                {
                    var img2 = kvList[j].Key;
                    if (moved.ContainsKey(img2)) continue;

                    double similarity = CosineSimilarity(kvList[i].Value, kvList[j].Value);
                    if (similarity > SimilarityThreshold)
                    {
                        // Try to mark img2 as moved atomically
                        if (moved.TryAdd(img2, true))
                        {
                            string dest = Path.Combine(duplicateFolder, Path.GetFileName(img2));
                            try
                            {
                                File.Move(img2, dest, true);

                                if (DeleteFromSource)
                                {
                                    File.Delete(img2);
                                }

                                Console.WriteLine($"Moved duplicate: {img2} (Similarity: {similarity:F2})");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Failed moving {img2}: {ex.Message}");
                            }
                        }
                    }
                }
            });

            Console.WriteLine("Processing complete!");
            OpenOutputDirectory(duplicateFolder);

            Console.ReadLine();
        }

        private static void OpenOutputDirectory(string duplicateFolder)
        {
            // Attempt to open the duplicateFolder in the platform's file explorer
            try
            {
                Console.WriteLine($"Opening duplicate folder: {duplicateFolder}");
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    // On Windows, UseShellExecute = true with the folder path opens File Explorer
                    Process.Start(new ProcessStartInfo { FileName = duplicateFolder, UseShellExecute = true });
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    // macOS 'open' command
                    Process.Start("open", duplicateFolder);
                }
                else
                {
                    // Linux (and other UNIX) 'xdg-open' command
                    Process.Start("xdg-open", duplicateFolder);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to open folder: {ex.Message}");
            }
        }

        static void PrintUsage()
        {
            Console.WriteLine("Usage:");
            Console.WriteLine("  --input-directory <path>       (required) directory containing images");
            Console.WriteLine("  --output-directory <path>      (optional) where to move duplicates; default: temp/Duplicates");
            Console.WriteLine("  --try-model <path>             (optional) path to ONNX model; default: Models\\resnet50-v2-7.onnx");
            Console.WriteLine("  --threshold <double>           (optional) similarity threshold (0-1); default: 0.95");
            Console.WriteLine("  --delete-from-source <bool>    (optional) delete duplicate from source after moving; default: false");
            Console.WriteLine("  --help, -h                     show this help");
        }

        static async Task<float[]> GetImageEmbeddingAsync(InferenceSession session, string imagePath, SemaphoreSlim sessionLock)
        {
            using var image = await Image.LoadAsync<Rgb24>(imagePath);
            image.Mutate(x => x.Resize(new Size(224, 224)));

            var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    var pixel = image[x, y];
                    inputTensor[0, 0, y, x] = pixel.R / 255f;
                    inputTensor[0, 1, y, x] = pixel.G / 255f;
                    inputTensor[0, 2, y, x] = pixel.B / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("data", inputTensor)
            };

            await sessionLock.WaitAsync();
            try
            {
                using var results = session.Run(inputs);
                var output = results.First().AsEnumerable<float>().ToArray();
                return Normalize(output);
            }
            finally
            {
                sessionLock.Release();
            }
        }

        static float[] Normalize(float[] vector)
        {
            float norm = (float)Math.Sqrt(vector.Sum(v => v * v));
            if (norm == 0) return vector;
            return vector.Select(v => v / norm).ToArray();
        }

        static double CosineSimilarity(float[] a, float[] b)
        {
            double dot = 0;
            for (int i = 0; i < a.Length; i++)
                dot += a[i] * b[i];
            return dot;
        }

        static bool IsImageFile(string path)
        {
            var ext = Path.GetExtension(path);
            if (string.IsNullOrEmpty(ext)) return false;
            switch (ext.ToLowerInvariant())
            {
                case ".jpg":
                case ".jpeg":
                case ".png":
                case ".bmp":
                case ".gif":
                case ".tiff":
                case ".webp":
                    return true;
                default:
                    return false;
            }
        }
    }
}
