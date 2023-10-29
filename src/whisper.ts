import * as ort from "onnxruntime-web";
import { ReadableStream, TransformStream, Transformer } from "node:stream/web";

// This resolves a bug with WASM in nodejs.
ort.env.wasm.numThreads = 1;
ort.env.remoteModels = false;

// wrapper around onnxruntime and model
class Whisper {
  private session: ort.InferenceSession;
  private minLength: Int32Array;
  private maxLength: Int32Array;
  private numReturnSequences: Int32Array;
  private lengthPenalty: Float32Array;
  private repetitionPenalty: Float32Array;
  private sampleRate: number;

  private constructor(session: ort.InferenceSession, sampleRate: number) {
    // semi constants that we initialize once and pass to every run() call
    this.session = session;
    this.sampleRate = sampleRate;
    this.minLength = Int32Array.from({ length: 1 }, () => 1);
    this.maxLength = Int32Array.from({ length: 1 }, () => 448);
    this.numReturnSequences = Int32Array.from({ length: 1 }, () => 1);
    this.lengthPenalty = Float32Array.from({ length: 1 }, () => 1);
    this.repetitionPenalty = Float32Array.from({ length: 1 }, () => 1);
  }

  /**
   * A factory method used to generate instances of whisper sessions.
   *
   * @param uri - The URL of PATH where the whisper model should be loaded from.
   * @returns - An initilized instance of the Whisper model for transcription.
   */
  static async create(
    uri: string = "whisper_cpu_int8_0_model.onnx",
    sampleRate: number
  ) {
    const opt: ort.InferenceSession.SessionOptions = {
      executionProviders: ["wasm"],
      logSeverityLevel: 3,
      logVerbosityLevel: 3,
    };

    const session = await ort.InferenceSession.create(uri, opt);

    return new Whisper(session, sampleRate);
  }

  /**
   * Converts a readable stream of raw audio channel data to text.
   *
   * @param audio - A stream of decoded PCM data chunks from an audio source.
   * @returns - A readable stream that contains a transformer used to transcribe audio to text.
   */
  async transcribe(audio: ReadableStream<Float32Array>) {
    const transformer: Transformer = {
      transform: async (chunk, controller) => {
        // Transform our tensor in the model.
        const value = await this.processAudio(chunk);

        // Enqueue our transformed text back into a stream.
        controller.enqueue(value);
      },
    };

    const transformStream = new TransformStream(transformer);

    return audio.pipeThrough(transformStream);
  }

  /**
   * Processes a chunk of raw decoded audio in steps, this funciton
   * is recursive and will keep processsing audio until the data
   * is empty. The results of these recursive calls are combined
   * into a single transcribed string.
   *
   * @param audio - Float32Array of a decoded audio stream.
   * @param idx - The starting position for transcription.
   * @param pos - The current position of a transcription.
   * @param results - The audio transcribed as a string.
   * @returns - A string containing all transcribed audio as text.
   */
  private async processAudio(
    audio: Float32Array,
    idx: number = 0,
    pos: number = 0,
    results = ""
  ): Promise<string> {
    if (idx < audio.length) {
      const ksteps = this.sampleRate * 30;
      const xa = audio.slice(idx, idx + ksteps);

      // Transform our tensor in the model.
      const transcribed = await this.run(new ort.Tensor(xa, [1, xa.length]));

      results = results + transcribed.str.data[0];

      // Return the result of the recursive call
      return await this.processAudio(audio, idx + ksteps, pos + 30, results);
    } else {
      return results;
    }
  }

  private async run(audio_pcm: ort.Tensor, beams = 1) {
    // Snake case is required by the onnyx runtime.
    const feed = {
      audio_pcm: audio_pcm,
      max_length: new ort.Tensor(new Int32Array(this.maxLength), [1]),
      min_length: new ort.Tensor(new Int32Array(this.minLength), [1]),
      num_beams: new ort.Tensor(
        Int32Array.from({ length: 1 }, () => beams),
        [1]
      ),
      num_return_sequences: new ort.Tensor(
        new Int32Array(this.numReturnSequences),
        [1]
      ),
      length_penalty: new ort.Tensor(new Float32Array(this.lengthPenalty), [1]),
      repetition_penalty: new ort.Tensor(
        new Float32Array(this.repetitionPenalty),
        [1]
      ),
    };

    return this.session.run(feed);
  }
}

export { Whisper };
