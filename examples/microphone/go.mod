module github.com/stanislaw-glogowski/openwakeword_go/examples/microphone

go 1.22

require (
	github.com/gordonklaus/portaudio v0.0.0-20260203164431-765aa7dfa631
	github.com/stanislaw-glogowski/openwakeword_go v0.0.0
	github.com/yalue/onnxruntime_go v1.31.0
)

replace github.com/stanislaw-glogowski/openwakeword_go => ../..
