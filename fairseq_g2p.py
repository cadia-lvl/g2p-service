# Copyright (c) Judy Y. Fong <lvl@judyyfong.xyz>
#

import fairseq
import torch
from fairseq.models.transformer import TransformerModel
import os

# The en2de and en2fr examples are licensed under the MIT License from
# pytorch/fairseq repo
# List available models
class Fairseq_g2p_transformer():
    def __init__(self, modelFile="final.mdl", encoding="UTF-8"):
        super(Options, self).__init__(modelFile=modelFile, encoding=encoding)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value

# class fairseq_g2p():
#     # def __init__()
    
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

# Select the dialect
def set_fairseq_paths(dialect='standard'):
    """ Select the paths based on dialect """
    modelDir = os.getenv("G2P_MODEL_DIR", "/data/models/g2p/fairseq/")
    possible_dialects = ['standard', 'north' , 'north_east', 'south']
    if dialect in possible_dialects:
        return (modelDir + '/data-bin/' + dialect, modelDir +
        '/checkpoints/' + dialect + '-256-.3-s-s/checkpoint_last.pt')
    else:
        raise ValueError("There is no matching dialect g2p model.")


def g2p_fairseq(words, dialect='standard'):
    """ Load the correct model and give a correct phonemes """
    # This g2p source code is licensed under the GPL-2.0 License found in the LICENSE
    # file in the root directory of this source tree.
    data_dir, checkpoint_file = set_fairseq_paths(dialect)
    g2p_lstm_model = TransformerModel.from_pretrained(data_dir, checkpoint_file)
    # Batched translation
    return g2p_lstm_model.translate(words)

# Function to change 'hlaupa' to 'h l a u p a' etc
def words2spaced(normal_words):
    """ 
    Change normal words to words with spaces between letters

         e.g. hlaupa to h l a u p a
    """
    separated = []
    for word in normal_words:
        separated.append(' '.join(char for char in word))
    return separated

def main():
    torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

    # TODO: callable from app.py

    # Process phrase to work with g2p functioon
    # TODO: remove punctuation because it affects the output
    # phrase = 'Velkomin til íslands.'
    # phrase = 'Velkomin til íslands'
    phrase = 'What is up Charlie Zinger Queen'
    # Change a phrase to a list of words with .split()
    phrase_spaced = words2spaced(phrase.split())

    # Process words to work with g2p function
    h_l_a_u_p_a = words2spaced(['hlaupa'])
    processed = words2spaced(
        ['Hlaupa', 'derp', 'orð', 'hrafn', 'daginn', 'Akureyri', 'banki']
    )

    # works with c, w, q, and z
    # g2p works with lowercased and capital letters
    # NOTE: punctuation just gives random output so shouldn't allow it to be
    # passed to g2p_fairseq
    print(g2p_fairseq(h_l_a_u_p_a))
    # ['l_0 9i: p a']
    print(g2p_fairseq(processed))
    # ['l_0 9i: p a', 't E r_0 p']
    print(g2p_fairseq(phrase_spaced))
    # ['c E l k_h O m I n', 't_h I: l', 'i s t l a n t s']

    print('\nnorth')
    print(g2p_fairseq(processed, 'north'))
    print('\nnorth east')
    print(g2p_fairseq(processed, 'north_east'))
    print('\nsouth')
    print(g2p_fairseq(processed, 'south'))

# ['hlaupa','orð', 'derp']
def fs_g2p_translation(word_list):
    """ Take in a normal word list and return pronunciation objects """
    w_o_r_d_l_i_s_t = words2spaced(word_list)
    word_phones = g2p_fairseq(w_o_r_d_l_i_s_t)
    fairseq_response = []
    for (phones, word) in zip(word_phones, word_list):
        fairseq_response.append({
            "word": word,
            "results": [ 
                { "pronunciation": phones }
            ] 
        })
    return fairseq_response    

if __name__ == '__main__':
    # fs_g2p_translation(word_list)
    main()
