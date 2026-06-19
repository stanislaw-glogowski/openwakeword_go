package openwakeword

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type onnxSession struct {
	session *ort.DynamicAdvancedSession
	inputs  []ort.InputOutputInfo
	outputs []ort.InputOutputInfo
}

func newONNXSession(path string) (*onnxSession, error) {
	inputs, outputs, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, fmt.Errorf("inspect ONNX model %q: %w", path, err)
	}
	inputNames := make([]string, len(inputs))
	outputNames := make([]string, len(outputs))
	for i := range inputs {
		inputNames[i] = inputs[i].Name
	}
	for i := range outputs {
		outputNames[i] = outputs[i].Name
	}
	session, err := ort.NewDynamicAdvancedSession(path, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("load ONNX model %q: %w", path, err)
	}
	return &onnxSession{session: session, inputs: inputs, outputs: outputs}, nil
}

func (s *onnxSession) runFloat(shape []int64, data []float32) (_ []float32, _ []int64, err error) {
	if len(s.inputs) != 1 || len(s.outputs) == 0 {
		return nil, nil, fmt.Errorf("expected one input and at least one output, got %d and %d", len(s.inputs), len(s.outputs))
	}
	input, err := ort.NewTensor(ort.NewShape(shape...), data)
	if err != nil {
		return nil, nil, err
	}
	defer func(input *ort.Tensor[float32]) {
		if deferErr := input.Destroy(); deferErr != nil {
			err = errors.Join(err, deferErr)
		}
	}(input)
	outputs := make([]ort.Value, len(s.outputs))
	if err := s.session.Run([]ort.Value{input}, outputs); err != nil {
		destroyONNXValues(outputs)
		return nil, nil, err
	}
	defer destroyONNXValues(outputs)
	output, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("expected float32 output, got %T", outputs[0])
	}
	return append([]float32(nil), output.GetData()...), append([]int64(nil), output.GetShape()...), nil
}

func (s *onnxSession) close() error {
	if s == nil || s.session == nil {
		return nil
	}
	err := s.session.Destroy()
	s.session = nil
	return err
}

func destroyONNXValues(values []ort.Value) {
	for _, value := range values {
		if value != nil {
			_ = value.Destroy()
		}
	}
}
