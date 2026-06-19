# `openwakeword_go`

Native Go implementation of the openWakeWord inference pipeline. It runs ONNX
melspectrogram, speech embedding, wake-word, and optional Silero VAD models
without Python.

## Features

- Streaming 16 kHz, mono, signed 16-bit PCM input
- Arbitrary input chunk sizes with 1280-sample internal framing
- Multiple binary or multiclass wake-word ONNX models
- Python-compatible startup suppression, patience, debounce, and VAD behavior
- Optional class mappings, verifier interface, audio filter interface, and WAV processing
- Direct integration with `github.com/yalue/onnxruntime_go`

## Requirements

The ONNX backend uses
[`github.com/yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go).
Callers own the global ONNX Runtime environment: initialize it before creating
an `Engine`, and destroy it after all engines and VAD instances are closed.

Your application must provide a compatible native ONNX Runtime shared library:

- macOS: `libonnxruntime.dylib`
- Linux: `libonnxruntime.so`
- Windows: `onnxruntime.dll`

Official openWakeWord model weights are licensed separately under
CC BY-NC-SA 4.0. The Go source in this repository is Apache-2.0.

## Setup Scripts

The repository includes small helper scripts for local development and the
microphone demo:

```bash
./scripts/download-models.sh
./scripts/download-runtime.sh
```

`download-models.sh` downloads the default openWakeWord ONNX files into
`./models`. `download-runtime.sh` detects the current platform, downloads ONNX
Runtime, and copies the shared library into `./runtime`.

Both scripts accept an optional output directory:

```bash
./scripts/download-models.sh /path/to/models
./scripts/download-runtime.sh /path/to/runtime
```

## Usage

```go
package main

import (
	"log"
	"time"

	openwakeword "github.com/stanislaw-glogowski/openwakeword_go"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	ort.SetSharedLibraryPath("runtime/libonnxruntime.dylib")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment()

	engine, err := openwakeword.New(openwakeword.Options{
		MelspectrogramModelPath: "models/melspectrogram.onnx",
		EmbeddingModelPath:      "models/embedding_model.onnx",
		VADModelPath:            "models/silero_vad.onnx",
		VADThreshold:            0.5,
		WakeWordModels: []openwakeword.ModelConfig{{
			Name: "hey_jarvis_v0.1",
			Path: "models/hey_jarvis_v0.1.onnx",
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	// Feed successive microphone frames. 1280 samples is 80 ms at 16 kHz.
	scores, err := engine.Predict(make([]int16, openwakeword.FrameSamples))
	if err != nil {
		log.Fatal(err)
	}
	if scores["hey_jarvis_v0.1"] >= 0.5 {
		log.Println("wake word detected")
	}

	_, _ = engine.PredictWithOptions(make([]int16, openwakeword.FrameSamples), openwakeword.PredictOptions{
		Threshold:    map[string]float32{"hey_jarvis_v0.1": 0.5},
		DebounceTime: 1250 * time.Millisecond,
	})
}
```

For a WAV file:

```go
predictions, err := engine.PredictWAV(
	"recording.wav", openwakeword.FrameSamples, time.Second,
	openwakeword.PredictOptions{},
)
```

The WAV must be mono 16-bit PCM at 16 kHz.

## Model Config

`ModelConfig.Name` controls the key returned by `Predict` for binary models.
If it is empty, the filename without `.onnx` is used.

For multiclass models, use `ClassMapping` to map output indexes to result keys:

```go
openwakeword.ModelConfig{
	Path: "models/multiclass.onnx",
	Name: "wakewords",
	ClassMapping: map[int]string{
		0: "alexa",
		1: "hey_jarvis_v0.1",
	},
}
```

`AudioFilter` can preprocess incoming PCM before feature extraction.
`Verifier` can run a second-stage score for a model after the first wake-word
score reaches `VerifierThreshold`.

## Tests

```bash
go test ./...
go vet ./...
```

The end-to-end test is opt-in because it needs native ONNX Runtime and model
files. The test initializes `github.com/yalue/onnxruntime_go` directly:

```bash
ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so \
OPENWAKEWORD_MODEL_DIR=/path/to/models \
OPENWAKEWORD_TEST_WAV=/path/to/alexa_test.wav \
go test -run TestOfficialONNXModels -v
```

`OPENWAKEWORD_MODEL_DIR` should contain:

- `melspectrogram.onnx`
- `embedding_model.onnx`
- `silero_vad.onnx`
- one or more wake-word models, for example `alexa_v0.1.onnx`

## Microphone Demo

The demo in `cmd/openwakeword_demo/main.go` listens to the default PortAudio
input. It reads configuration from a working directory, using `--cwd` or the
current directory by default.

Use the setup scripts above to download the default model files and ONNX
Runtime library.

Expected layout:

```text
.
├── models
│   └── hey_jarvis_v0.1.onnx
│   ├── alexa_v0.1.onnx
│   ├── embedding_model.onnx
│   ├── melspectrogram.onnx
│   ├── silero_vad.onnx
└── runtime
    └── libonnxruntime.dylib
```

On Linux the runtime file is `runtime/libonnxruntime.so`; on Windows it is
`runtime/onnxruntime.dll`.

Every additional `models/*.onnx` file is treated as a wake-word model unless it
is one of the feature/VAD files above. The result name is the filename without
`.onnx`, for example `alexa_v0.1.onnx` becomes `alexa_v0.1`.

On macOS, install the native PortAudio library first:

```bash
brew install portaudio
```

Run the demo:

```bash
go run ./cmd/openwakeword_demo
```

Useful flags:

```bash
go run ./cmd/openwakeword_demo \
  --cwd /path/to/openwakeword_assets \
  --threshold 0.9 \
  --vad-threshold 0.5 \
  --warmup 3s \
  --debounce 1500ms
```

Set `--vad-threshold 0` to disable VAD. The default microphone must support
mono capture at 16 kHz. macOS will ask for microphone permission the first time
the program starts recording.
