from typing import Any, Dict

import argparse
import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from laion_clap import CLAP_Module
from transformers import AutoTokenizer
from sentence_transformers import util
from loss import OTLossAlign, OTBaseline, OTLossKernel


from modeling.enclap_bart import EnClapBartConfig, EnClapBartForConditionalGeneration


class EnClap:
    def __init__(
        self,
        ckpt_path: str,
        clap_audio_model: str = "HTSAT-tiny",
        clap_enable_fusion: bool = True,
        clap_ckpt_path: str = None,
        device: str = "cuda",
    ):
        config = EnClapBartConfig.from_pretrained(ckpt_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        self.model = (
            EnClapBartForConditionalGeneration.from_pretrained(ckpt_path)
            .to(self.device)
            .eval()
        )

        self.encodec = EncodecModel.encodec_model_24khz().to(self.device)
        self.encodec.set_target_bandwidth(12.0)
        self.clap_model = CLAP_Module(enable_fusion=clap_enable_fusion, amodel=clap_audio_model, device=self.device)
        self.clap_model.load_ckpt(clap_ckpt_path)

        self.generation_config = {
            "_from_model_config": True,
            "bos_token_id": 0,
            "decoder_start_token_id": 2,
            "early_stopping": True,
            "eos_token_id": 2,
            "forced_bos_token_id": 0,
            "forced_eos_token_id": 2,
            "no_repeat_ngram_size": 3,
            # "num_beams": 4,
            "pad_token_id": 1,
            "max_length": 50,
            # "top_p": 0.9,
            "temperature": 1.5,
            # "top_k": 6,
            "num_return_sequences":20,
            "do_sample": True,
            "output_scores": True,
            "return_dict_in_generate" : True
            # "output_hidden_states": True
            # "output_attentions" : True
        }
        self.max_seq_len = config.max_position_embeddings - 3

    @torch.no_grad()
    def infer_from_audio_file(
        self, audio_file: str, generation_config: Dict[str, Any] = None
    ) -> str:
        if generation_config is None:
            generation_config = self.generation_config
        audio, res = torchaudio.load(audio_file)
        return self.infer_from_audio(audio[0], res)

    @torch.no_grad()
    def infer_from_audio(
        self, audio: torch.Tensor, res: int, generation_config: Dict[str, Any] = None
    ) -> str:
        if generation_config is None:
            generation_config = self.generation_config
        if audio.dtype == torch.short:
            audio = audio / 2**15
        if audio.dtype == torch.int:
            audio = audio / 2**31
        encodec_audio = (
            convert_audio(
                audio.unsqueeze(0), res, self.encodec.sample_rate, self.encodec.channels
            )
            .unsqueeze(0)
            .to(self.device)
        )
        encodec_frames = self.encodec.encode(encodec_audio)
        encodec_frames = torch.cat(
            [codebook for codebook, _ in encodec_frames], dim=-1
        ).mT

        clap_audio = torchaudio.transforms.Resample(res, 48000)(audio).unsqueeze(0)
        clap_embedding = self.clap_model.get_audio_embedding_from_data(clap_audio, use_tensor=True)

        return self._infer(encodec_frames, clap_embedding, generation_config)

    @torch.no_grad()
    def _infer_org(
        self,
        encodec_frames: torch.LongTensor,
        clap_embedding: torch.Tensor,
        generation_config: Dict[str, Any] = None,
    ) -> str:
        input_ids = torch.cat(
            [
                torch.ones(
                    (encodec_frames.shape[0], 2, encodec_frames.shape[-1]),
                    dtype=torch.long,
                ).to(self.device)
                * self.tokenizer.bos_token_id,
                encodec_frames[:, : self.max_seq_len],
                torch.ones(
                    (encodec_frames.shape[0], 1, encodec_frames.shape[-1]),
                    dtype=torch.long,
                ).to(self.device)
                * self.tokenizer.eos_token_id,
            ],
            dim=1,
        )
        encodec_mask = torch.LongTensor(
            [[0, 0] + [1] * (input_ids.shape[1] - 3) + [0]]
        ).to(self.device)

        enclap_bart_inputs = {
            "input_ids": input_ids,
            "encodec_mask": encodec_mask,
            "clap_embedding": clap_embedding,
        }

        results = self.model.generate(**enclap_bart_inputs, **generation_config)
        caption = self.tokenizer.batch_decode(results, skip_special_tokens=True)

        return caption
    ### infer cosine
    @torch.no_grad()
    def _infer_cosine(
        self,
        encodec_frames: torch.LongTensor,
        clap_embedding: torch.Tensor,
        generation_config: Dict[str, Any] = None,
        alpha: float = 0.5
    ) -> str:
        input_ids = torch.cat(
            [
                torch.ones(
                    (encodec_frames.shape[0], 2, encodec_frames.shape[-1]),
                    dtype=torch.long,
                ).to(self.device)
                * self.tokenizer.bos_token_id,
                encodec_frames[:, : self.max_seq_len],
                torch.ones(
                    (encodec_frames.shape[0], 1, encodec_frames.shape[-1]),
                    dtype=torch.long,
                ).to(self.device)
                * self.tokenizer.eos_token_id,
            ],
            dim=1,
        )
        encodec_mask = torch.LongTensor(
            [[0, 0] + [1] * (input_ids.shape[1] - 3) + [0]]
        ).to(self.device)

        enclap_bart_inputs = {
            "input_ids": input_ids,
            "encodec_mask": encodec_mask,
            "clap_embedding": clap_embedding,
        }

        results = self.model.generate(**enclap_bart_inputs, **generation_config)
        num_samp_sen = results.sequences.shape[0]

        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        probs = results.sequences_scores.exp()
        encoder_outputs = self.model.model.encoder(**enclap_bart_inputs)

        encoder_hidden = encoder_outputs[0]*encodec_mask.unsqueeze(-1)
        encoder_hidden = encoder_hidden.sum(dim=1)/encodec_mask.squeeze(-1).sum(-1, keepdim=True)
        # print("results sequences: ",results.sequences.shape)
        # print("encoder output: ", encoder_outputs[0].shape)
        decoder_attn_mask = results.sequences.eq(1)
        decoder_attn_mask = ~decoder_attn_mask

        decoder_outputs = self.model(
            labels=results.sequences,
            input_ids=input_ids.expand(num_samp_sen,-1,-1),
            encodec_mask=encodec_mask.expand(num_samp_sen,-1),
            clap_embedding=clap_embedding.expand(num_samp_sen, -1)
            # encoder_hidden_states=encoder_outputs[0].expand(10, -1, -1)
        )
        decoder_hidden = decoder_outputs[0]*decoder_attn_mask.unsqueeze(-1)
        decoder_hidden = decoder_hidden.sum(dim=1)/decoder_attn_mask.squeeze(-1).sum(-1, keepdim=True)
        # print("decoder hidden shape: ", decoder_hidden[0].shape)
        # print("decoder mask: ", decoder_attn_mask)
        similarity = cos(encoder_hidden, decoder_hidden)
        score = (1-alpha)*probs + alpha*similarity
        max_ind = torch.argmax(score)

        caption = self.tokenizer.batch_decode([results.sequences[max_ind]], skip_special_tokens=True)
        prob_cap = self.tokenizer.batch_decode([results.sequences[0]], skip_special_tokens=True)

        return caption
    
    @torch.no_grad()
    def _infer(
        self,
        encodec_frames: torch.LongTensor,
        clap_embedding: torch.Tensor,
        generation_config: Dict[str, Any] = None,
        alpha: float = 0.5
    ) -> str:
        input_ids = torch.cat(
            [
                torch.ones(
                    (encodec_frames.shape[0], 2, encodec_frames.shape[-1]),
                    dtype=torch.long,
                ).to(self.device)
                * self.tokenizer.bos_token_id,
                encodec_frames[:, : self.max_seq_len],
                torch.ones(
                    (encodec_frames.shape[0], 1, encodec_frames.shape[-1]),
                    dtype=torch.long,
                ).to(self.device)
                * self.tokenizer.eos_token_id,
            ],
            dim=1,
        )
        encodec_mask = torch.LongTensor(
            [[0, 0] + [1] * (input_ids.shape[1] - 3) + [0]]
        ).to(self.device)

        enclap_bart_inputs = {
            "input_ids": input_ids,
            "encodec_mask": encodec_mask,
            "clap_embedding": clap_embedding,
        }
        ot_loss = OTLossKernel(use_pos=True, pos_dim=1024)
        # ot_loss = OTLossAlign(use_pos=True, pos_dim=1024)
        # ot_loss = OTBaseline()
        # ot_loss = OTLossAlignTime()
        # ot_loss = OTEnergyLossAlign(pos_dim=1024)
        results = self.model.generate(**enclap_bart_inputs, **generation_config)
        num_samp_sen = results.sequences.shape[0]

        probs = results.sequences_scores.exp()
        encoder_outputs = self.model.model.encoder(**enclap_bart_inputs)

        encoder_hidden = encoder_outputs[0]*encodec_mask.unsqueeze(-1)
        encode_masks = encodec_mask

        decoder_attn_mask = results.sequences.eq(1)
        decoder_attn_mask = ~decoder_attn_mask

        decoder_outputs = self.model(
            labels=results.sequences,
            input_ids=input_ids.expand(num_samp_sen,-1,-1),
            encodec_mask=encodec_mask.expand(num_samp_sen,-1),
            clap_embedding=clap_embedding.expand(num_samp_sen, -1)
            # encoder_hidden_states=encoder_outputs[0].expand(10, -1, -1)
        )
        decoder_hidden = decoder_outputs.decoder_hidden_states*decoder_attn_mask.unsqueeze(-1)
        decode_mask = decoder_attn_mask
        # print("encode masks: ", encode_masks.shape)
        # print("decoder masks: ", decode_mask.shape)
        # print("encoder hidden shape: ", encoder_hidden.shape)
        # print("decoder hidden shape: ", decoder_outputs.decoder_hidden_states.shape)

        # similarity = cos(encoder_hidden, decoder_hidden)
        similarity = torch.zeros(num_samp_sen).to(probs.device)
        for i in range(num_samp_sen):
            dist = ot_loss(encoder_hidden, encode_masks, decoder_hidden[i].unsqueeze(0), decode_mask[i].unsqueeze(0))
            # similarity[i] = 1-dist
            similarity = dist
        # print("Sim: ", similarity)
        # print("probs: ", probs)
        # print("*"*90)
        score = (1-alpha)*probs + alpha*similarity
        max_ind = torch.argmax(score)
        max_prob_ind = torch.argmax(probs)

        caption = self.tokenizer.batch_decode([results.sequences[max_ind]], skip_special_tokens=True)
        max_prob_cap =  self.tokenizer.batch_decode([results.sequences[max_prob_ind]], skip_special_tokens=True)
        prob_cap = self.tokenizer.batch_decode([results.sequences[0]], skip_special_tokens=True)
        # print("max score cap: ", caption)
        # print("max prob cap: ", max_prob_cap)
        # print("*"*90)

        return caption

    @torch.no_grad()
    def infer_from_encodec(
        self,
        encodec_path,
        clap_path,
        generation_config: Dict[str, Any] = None,
    ):
        if generation_config is None:
            generation_config = self.generation_config
        encodec_frames = torch.from_numpy(np.load(encodec_path)).unsqueeze(0).cuda()
        clap_embedding = torch.from_numpy(np.load(clap_path)).unsqueeze(0).cuda()

        return self._infer(encodec_frames, clap_embedding, generation_config)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", type=str)
    parser.add_argument("--clap_ckpt", '-cl', type=str)
    parser.add_argument("--input", "-i", type=str)
    args = parser.parse_args()

    print("> Loading Model...")
    enclap = EnClap(
        ckpt_path=args.ckpt, 
        clap_ckpt_path=args.clap_ckpt, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("> Running Inference...")
    prediction = enclap.infer_from_audio_file(args.input)[0]
    print("> Result: ", prediction)