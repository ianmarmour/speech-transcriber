# speech-transcriber

NodeJS library providing access to OpenAI's Whisper model.

## Install

```bash
npm install --save "speech-transcriber"
```

## Usage

```ts
// Create a new instance of the Transcriber
// for use with 16000hz audio data.
const transcriber = await Transcriber.create(16000);

// Create a transcription stream for reading text
const stream = await transcriber.transcribe(speechSegmentStream);

// Iterate over chunks of transcription stream
// each chunk contains a string that was transcribed
// from a Float32Array.
for await (const chunk of stream) {
  console.log("Received transcribed text:", chunk);
}
```
