"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import json
from typing import Dict, Union

import gradio as gr
import numpy as np
import torch
import torchaudio
from attrdict import AttrDict
from diffusion.respace import SpacedDiffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from model.diffusion import FiLMTransformer
from utils.misc import fixseed
from utils.model_util import create_model_and_diffusion, load_model
from visualize.render_codes import BodyTransformer


class PoseModel:
    def __init__(self, pose_args) -> None:
        self.pose_model, self.pose_diffusion, self.device = self._setup_model(
            pose_args, "checkpoints/diffusion/c1_pose/model000340000.pt"
        )
        # load standardization stuff
        stats = torch.load("dataset/PXB184/data_stats.pth")
        stats["pose_mean"] = stats["pose_mean"].reshape(-1)
        stats["pose_std"] = stats["pose_std"].reshape(-1)
        self.stats = stats
        config_base = f"./checkpoints/ca_body/data/PXB184"
        self.body_transformer = BodyTransformer(
            config_base=config_base,
        )

    def _setup_model(
        self,
        args_path: str,
        model_path: str,
    ) -> (Union[FiLMTransformer, ClassifierFreeSampleModel], SpacedDiffusion):
        with open(args_path) as f:
            args = json.load(f)
        args = AttrDict(args)
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("running on...", args.device)
        args.model_path = model_path
        args.output_dir = "/tmp/gradio/"
        args.timestep_respacing = "ddim100"
        if args.data_format == "pose":
            args.resume_trans = "checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt"

        ## create model
        model, diffusion = create_model_and_diffusion(args, split_type="test")
        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location=args.device)
        load_model(model, state_dict)
        model = ClassifierFreeSampleModel(model)
        model.eval()
        model.to(args.device)
        print("Loaded model!")
        return model, diffusion, args.device

    def _replace_keyframes(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        B: int,
        T: int,
        top_p: float = 0.97,
    ) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.pose_model.transformer.generate(
                model_kwargs["y"]["audio"],
                T,
                layers=self.pose_model.tokenizer.residual_depth,
                n_sequences=B,
                top_p=top_p,
            )
        tokens = tokens.reshape((B, -1, self.pose_model.tokenizer.residual_depth))
        pred = self.pose_model.tokenizer.decode(tokens).detach()
        return pred

    def _run_single_diffusion(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        diffusion: SpacedDiffusion,
        model: Union[FiLMTransformer, ClassifierFreeSampleModel],
        curr_seq_length: int,
        num_repetitions: int = 1,
    ) -> (torch.Tensor,):
        sample_fn = diffusion.ddim_sample_loop
        with torch.no_grad():
            sample = sample_fn(
                model,
                (num_repetitions, model.nfeats, 1, curr_seq_length),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        return sample

    def generate_sequences(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        data_format: str,
        curr_seq_length: int,
        num_repetitions: int = 5,
        guidance_param: float = 10.0,
        top_p: float = 0.97,
        # batch_size: int = 1,
    ) -> Dict[str, np.ndarray]:
        if data_format == "pose":
            model = self.pose_model
            diffusion = self.pose_diffusion
        else:
            model = self.face_model
            diffusion = self.face_diffusion

        all_motions = []
        model_kwargs["y"]["scale"] = torch.ones(num_repetitions) * guidance_param
        model_kwargs["y"] = {
            key: val.to(self.device) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }
        if data_format == "pose":
            model_kwargs["y"]["mask"] = (
                torch.ones((num_repetitions, 1, 1, curr_seq_length))
                .to(self.device)
                .bool()
            )
            model_kwargs["y"]["keyframes"] = self._replace_keyframes(
                model_kwargs,
                num_repetitions,
                int(curr_seq_length / 30),
                top_p=top_p,
            )
        sample = self._run_single_diffusion(
            model_kwargs, diffusion, model, curr_seq_length, num_repetitions
        )
        all_motions.append(sample.cpu().numpy())
        print(f"created {len(all_motions) * num_repetitions} samples")
        return np.concatenate(all_motions, axis=0)

pose_model = PoseModel(
    pose_args="./checkpoints/diffusion/c1_pose/args.json",
)

def generate_results(audio: np.ndarray, num_repetitions: int, top_p: float):
    if audio is None:
        raise gr.Error("Please record audio to start")
    sr, y = audio
    # set to mono and perform resampling
    y = torch.Tensor(y)
    if y.dim() == 2:
        dim = 0 if y.shape[0] == 2 else 1
        y = torch.mean(y, dim=dim)
    y = torchaudio.functional.resample(torch.Tensor(y), orig_freq=sr, new_freq=48_000)
    sr = 48_000
    # make it so that it is 4 seconds long
    if len(y) < (sr * 4):
        raise gr.Error("Please record at least 4 second of audio")
    if num_repetitions is None or num_repetitions <= 0 or num_repetitions > 10:
        raise gr.Error(
            f"Invalid number of samples: {num_repetitions}. Please specify a number between 1-10"
        )
    cutoff = int(len(y) / (sr * 4))
    y = y[: cutoff * sr * 4]
    curr_seq_length = int(len(y) / sr) * 30
    # create model_kwargs
    model_kwargs = {"y": {}}
    dual_audio = np.random.normal(0, 0.001, (1, len(y), 2))
    dual_audio[:, :, 0] = y / max(y)
    dual_audio = (dual_audio - pose_model.stats["audio_mean"]) / pose_model.stats[
        "audio_std_flat"
    ]
    model_kwargs["y"]["audio"] = (
        torch.Tensor(dual_audio).float().tile(num_repetitions, 1, 1)
    )
    pose_results = (
        pose_model.generate_sequences(
            model_kwargs,
            "pose",
            curr_seq_length,
            num_repetitions=int(num_repetitions),
            guidance_param=2.0,
            top_p=top_p,
        )
        .squeeze(2)
        .transpose(0, 2, 1)
    )
    pose_results = (
        pose_results * pose_model.stats["pose_std"] + pose_model.stats["pose_mean"]
    )
    dual_audio = (
        dual_audio * pose_model.stats["audio_std_flat"]
        + pose_model.stats["audio_mean"]
    )
    return pose_results, dual_audio[0].transpose(1, 0).astype(np.float32)


def audio_to_avatar(audio: np.ndarray, num_repetitions: int, top_p: float):
    pose_results, audio = generate_results(audio, num_repetitions, top_p)
    # save to audio_i.wav
    torchaudio.save(f"audio_output.wav", torch.Tensor(audio), 48000) # [2, 192000]
    print("saved audio_output.wav")
    B = len(pose_results)
    results = []
    for i in range(B):
        # "body_motion": pose_results[i, ...],  # B(1) x T x 104
        # save body_motion_i.npy
        np.save(f"body_motion_{i}.npy", pose_results[i, ...])
        print(f"saved body_motion_{i}.npy")
        body_motions = pose_model.body_transformer.motion_transform(
            torch.Tensor(pose_results[i, ...])
        )
        print("body_motions shape:", body_motions.shape) # [2T, 159, 8]
        # save body_motions_i.npy
        np.save(f"body_motions_{i}.npy", body_motions)
    return results


demo = gr.Interface(
    audio_to_avatar,  # function
    [
        gr.Audio(sources=["microphone"]),
        gr.Number(
            value=1,
            label="Number of Samples (default = 1)",
            precision=0,
            minimum=1,
            maximum=10,
        ),
        gr.Number(
            value=0.97,
            label="Sample Diversity (default = 0.97)",
            precision=None,
            minimum=0.01,
            step=0.01,
            maximum=1.00,
        ),
    ],  # input type
    [gr.Video(format="mp4", visible=True)]
    + [gr.Video(format="mp4", visible=False) for _ in range(9)],  # output type
    title='"From Audio to Photoreal Embodiment: Synthesizing Humans in Conversations" Demo',
    description="You can generate a photorealistic avatar from your voice! <br/>\
        1) Start by recording your audio.  <br/>\
        2) Specify the number of samples to generate.  <br/>\
        3) Specify how diverse you want the samples to be. This tunes the cumulative probability in nucleus sampling: 0.01 = low diversity, 1.0 = high diversity.  <br/>\
        4) Then, sit back and wait for the rendering to happen! This may take a while (e.g. 30 minutes) <br/>\
        5) After, you can view the videos and download the ones you like.  <br/>",
    article="Relevant links: [Project Page](https://people.eecs.berkeley.edu/~evonne_ng/projects/audio2photoreal)",  # TODO: code and arxiv
)

if __name__ == "__main__":
    fixseed(10)
    demo.launch(share=True)
