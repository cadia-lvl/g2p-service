# Copyright (c) Judy Y. Fong <lvl@judyyfong.xyz>
#

import fairseq
import torch
from fairseq.models.transformer import TransformerModel

# The en2de and en2fr examples are licensed under the MIT License from
# pytorch/fairseq repo
# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

def en2de_example():
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

def en2fr_example():
    print("\nEnglish to French")
    en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr')
    # Translate a sentence
    print (en2fr.translate('Hello world!'))

def g2p_stan_example(words):
    # This g2p source code is licensed under the GPL-2.0 License found in the LICENSE
    # file in the root directory of this source tree.
    print("\n\nUsing the g2p models within python")
    common_path='/data/models/g2p/fairseq/'
    g2p_stan = TransformerModel.from_pretrained(common_path + 'data-bin/standard', 
        checkpoint_file=common_path + 'checkpoints/standard-256-.3-s-s/checkpoint_last.pt')
    # Batched translation
    print(g2p_stan.translate(words))

# Function to change 'hlaupa' to 'h l a u p a' etc
def words2spaced(normal_words):
    # TODO: check if it's words or one word
    separated = []
    for word in normal_words:
        separated.append(' '.join(char for char in word))
    return separated

def main():
    # TODO: add list the requirements necessary, put the models somewhere
    # TODO: work with all the dialects
    # TODO: callable from app.py
    # TODO: change filename

    # Process phrase to work with g2p functioon
    # TODO: remove punctuation because it affects the output
    # phrase = 'Velkomin til íslands.'
    # phrase = 'Velkomin til íslands'
    phrase = 'What is up Charlie Zinger Queen'
    # Change a phrase to a list of words with .split()
    phrase_spaced = words2spaced(phrase.split())

    # Process words to work with g2p function
    h_l_a_u_p_a = words2spaced(['hlaupa'])
    processed = words2spaced(['Hlaupa', 'derp', 'orð'])

    # works with c, w, q, and z
    # g2p works with lowercased and capital letters
    # NOTE: punctuation just gives random output so shouldn't use it
    g2p_stan_example(h_l_a_u_p_a)
    # ['l_0 9i: p a']
    g2p_stan_example(processed)
    # ['l_0 9i: p a', 't E r_0 p']
    g2p_stan_example(phrase_spaced)
    # ['c E l k_h O m I n', 't_h I: l', 'i s t l a n t s']

if __name__ == '__main__':
    main()
