# Copyright (c) Judy Y. Fong
#
# TODO: add list the requirements necessary, put the models somewhere
# TODO: callable from app.py

import fairseq
import torch
from fairseq.models.transformer import TransformerModel

# The en2de and en2fr examples are licensed under the MIT License from
# pytorch/fairseq repo
# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

print("English to German")
# Load a transformer trained on WMT'19 En-De
# Note: WMT'19 models use fastBPE instead of subword_nmt, see instructions below
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.eval()  # disable dropout
# The underlying model is available under the *models* attribute
assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)
# Translate a sentence
print (en2de.translate('Hello world!'))
# 'Hallo Welt!'
# Batched translation
print(en2de.translate(['Hello world!', 'The cat sat on the mat.']))
# ['Hallo Welt!', 'Die Katze saß auf der Matte.']

print("\nEnglish to French")
en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr')
# Translate a sentence
print (en2fr.translate('Hello world!'))

# This g2p source code is licensed under the GPL-2.0 License found in the LICENSE
# file in the root directory of this source tree.
print("\n\nUsing the g2p models within python")
common_path='/data/models/g2p/fairseq/'
g2p_stan = TransformerModel.from_pretrained(common_path + 'data-bin/standard', 
    checkpoint_file=common_path + 'checkpoints/standard-256-.3-s-s/checkpoint_last.pt')
# Translate a word
print(g2p_stan.translate('h l a u p a'))
# 'l_0 9i: p a'
# Batched translation
print(g2p_stan.translate(['h l a u p a','d e r p']))
# ['l_0 9i: p a', 't E r_0 p']
