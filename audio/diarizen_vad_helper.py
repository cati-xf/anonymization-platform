"""
DiariZen-based voice-activity-detection helper. Run in the diarizen conda environment
via subprocess, since DiariZen needs a different dependency set (pinned torch/CUDA,
a patched pyannote-audio fork) than the rest of this codebase.

Loads a pretrained DiariZen diarization model and writes the union of all speaker
segments (i.e. a speech/non-speech timeline, ignoring speaker identity) as a JSON
list of [start_sec, end_sec] pairs.

Usage:
    python diarizen_vad_helper.py \
        --audio noise.wav \
        --output segments.json \
        [--model BUT-FIT/diarizen-wavlm-large-s80-md-v2]
"""
import argparse
import json
import traceback

from diarizen.pipelines.inference import DiariZenPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    args = parser.parse_args()

    pipeline = DiariZenPipeline.from_pretrained(args.model)

    try:
        diarization = pipeline(args.audio)
        segments = [[segment.start, segment.end] for segment in diarization.get_timeline().support()]
    except ValueError:
        # DiariZen's clustering step can crash (rather than return an empty result)
        # when it finds no speech at all in the input — expected on a mostly-clean
        # noise stem. Treat that as "no leftover speech" instead of failing the job.
        traceback.print_exc()
        print("Assuming no speech present, writing empty segment list", flush=True)
        segments = []

    with open(args.output, "w") as f:
        json.dump(segments, f)
    print(f"Found {len(segments)} speech segment(s)", flush=True)


if __name__ == "__main__":
    main()
