package openwakeword

import (
	"errors"
	"fmt"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

type VAD struct {
	model     *onnxSession
	h         []float32
	c         []float32
	history   []float32
	threshold float32
}

func NewVAD(path string, threshold float32) (*VAD, error) {
	model, err := newONNXSession(path)

	if err != nil {
		return nil, fmt.Errorf("load VAD model: %w", err)
	}
	v := &VAD{model: model, threshold: threshold}
	v.Reset()
	return v, nil
}

func (v *VAD) Reset() {
	v.h = make([]float32, 2*64)
	v.c = make([]float32, 2*64)
	v.history = v.history[:0]
}

func (v *VAD) Predict(samples []int16, frameSize int) (float32, error) {
	if frameSize <= 0 {
		return 0, errors.New("VAD frame size must be positive")
	}
	if len(samples) == 0 {
		return 0, nil
	}
	var sum float32
	var chunks int
	for start := 0; start < len(samples); start += frameSize {
		end := start + frameSize
		if end > len(samples) {
			end = len(samples)
		}
		audio := make([]float32, end-start)
		for i, sample := range samples[start:end] {
			audio[i] = float32(sample) / 32767
		}
		inputs := make([]ort.Value, len(v.model.inputs))
		for i, input := range v.model.inputs {
			var value ort.Value
			var err error
			switch strings.ToLower(input.Name) {
			case "input":
				value, err = ort.NewTensor(ort.NewShape(1, int64(len(audio))), audio)
			case "h":
				value, err = ort.NewTensor(ort.NewShape(2, 1, 64), v.h)
			case "c":
				value, err = ort.NewTensor(ort.NewShape(2, 1, 64), v.c)
			case "sr":
				value, err = ort.NewTensor(ort.NewShape(1), []int64{SampleRate})
			default:
				return 0, fmt.Errorf("unsupported VAD input %q", input.Name)
			}
			if err != nil {
				destroyONNXValues(inputs)
				return 0, fmt.Errorf("create VAD input %q: %w", input.Name, err)
			}
			inputs[i] = value
		}
		outputs := make([]ort.Value, len(v.model.outputs))
		runErr := v.model.session.Run(inputs, outputs)
		destroyONNXValues(inputs)
		if runErr != nil {
			destroyONNXValues(outputs)
			return 0, fmt.Errorf("run VAD model: %w", runErr)
		}
		var score *float32
		for i, output := range outputs {
			lower := strings.ToLower(v.model.outputs[i].Name)
			tensor, ok := output.(*ort.Tensor[float32])
			if !ok {
				return 0, fmt.Errorf("unsupported VAD output %q type %T", v.model.outputs[i].Name, output)
			}
			data := tensor.GetData()
			switch {
			case lower == "hn" || lower == "h":
				v.h = append(v.h[:0], data...)
			case lower == "cn" || lower == "c":
				v.c = append(v.c[:0], data...)
			case len(data) > 0:
				value := data[0]
				score = &value
			}
		}
		if score == nil {
			destroyONNXValues(outputs)
			return 0, errors.New("VAD model did not return a score")
		}
		sum += *score
		chunks++
		destroyONNXValues(outputs)
	}
	average := sum / float32(chunks)
	v.history = append(v.history, average)
	if len(v.history) > 125 {
		v.history = v.history[len(v.history)-125:]
	}
	return average, nil
}

// ContextScore matches openWakeWord's delayed speech context: the maximum VAD
// score from frames 7 through 4 positions before the current prediction.
func (v *VAD) ContextScore() float32 {
	end := len(v.history) - 4
	start := len(v.history) - 7
	if start < 0 || end <= start {
		return 0
	}
	score := v.history[start]
	for _, s := range v.history[start+1 : end] {
		if s > score {
			score = s
		}
	}

	return score
}

func (v *VAD) CheckContextScore() bool {
	return v.ContextScore() >= v.threshold
}

func (v *VAD) Close() error {
	if v == nil || v.model == nil {
		return nil
	}
	err := v.model.close()
	v.model = nil
	return err
}
