Overview
=====

The Imaginary CLI Tool is designed to help you identify and manage duplicate image files within a specified directory, allowing you to reclaim valuable space on your local storage devices.

Note: This tool operates only on offline storage such as HDDs or SSDs. Cloud-based storage locations are not supported.


How It Works
=====

The tool leverages an AI-powered image embedding model to analyze and compare image files. It performs the following steps:

1. Embedding Generation – Each image is processed using a vision model that converts it into a numerical vector representation (embedding).

2. Duplicate Detection – The generated embeddings are compared to identify visually similar or duplicate images.

3. File Organization – Detected duplicates are automatically moved to a specified output directory for review or deletion.

Command-line parsing
--------------------
The program now uses command-line parsing instead of `Environment.GetEnvironmentVariable`. A helper `ParseArg` is available and behaves as follows:
- Supports both `--key=value` and `--key value` forms.
- Returns `null` if:
  - the key is not present, or
  - the key is present but has no value (e.g., `--key` at end of args or followed by another key).
- Use the parsed strings to convert to the required types (double and bool) using `CultureInfo.InvariantCulture`.

Recognized options and defaults
-------------------------------
- `--input-directory` (required)
  - Path to the directory containing files to process.
- `--output-directory` (optional)
  - Default: `Path.GetTempPath()` + Path separator + `Duplicates`
  - Example default value: `C:\Users\<User>\AppData\Local\Temp\Duplicates` on Windows.
- `--try-model` (optional)
  - Default: `Models/resnet50-v2-7.onnx`
  - Path to the ONNX model used for embedding generation.
- `--threshold` (optional)
  - Default: `0.95`
  - Parsed as `double` using `CultureInfo.InvariantCulture`.
- `--delete-from-source` (optional)
  - Default: `false`
  - Parsed as `bool` using `CultureInfo.InvariantCulture`. Accepts `true` / `false`.
- `--help` or `-h`
  - Print usage information and exit.

Validation rules
----------------
- `--input-directory` must be provided and must exist.
  - If missing or does not exist: program prints a usage error and exits with a non-zero status.
- `--try-model` must point to an existing file (if provided). If omitted, the default model path must exist.
  - If the model file does not exist: program prints an error and exits with a non-zero status.

Usage examples
--------------
1) Using `--key=value` forms:
   - Example:
     dotnet run -- --input-directory="C:\Data\Images" --output-directory="D:\Duplicates" --try-model="Models\resnet50-v2-7.onnx" --threshold=0.92 --delete-from-source=true

2) Using `--key value` forms:
   - Example:
     dotnet run -- --input-directory "C:\Data\Images" --output-directory "D:\Duplicates" --try-model "Models\resnet50-v2-7.onnx" --threshold 0.92 --delete-from-source true

3) Minimal required:
   - Example (uses defaults for everything else):
     dotnet run -- --input-directory "C:\Data\Images"

4) Help:
   - Example:
     dotnet run -- --help
     or
     dotnet run -- -h

Exit codes
----------
- Exit `0` on success or after printing `--help`.
- Exit non-zero on validation or runtime errors.

Disclaimer
----------
The owner of this tool is not responsible for any loss or corruption of files during the run. Use this tool at your own risk. Always back up important data before running file-manipulating tools.

Quick reference: sample help text
--------------------------------
--input-directory <path>     (required) Path to files to scan.
--output-directory <path>    (optional) Default: Path.GetTempPath()/Duplicates
--try-model <path>           (optional) Default: Models/resnet50-v2-7.onnx
--threshold <double>         (optional) Default: 0.95 (InvariantCulture)
--delete-from-source <bool>  (optional) Default: false (InvariantCulture)
--help, -h                   Show this help and exit.
