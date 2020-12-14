import os
import math
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from g2p import SequiturTool, Translator, loadG2PSample
from fairseq_g2p import FairseqGraphemeToPhoneme as fs_g2p

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app)

# TODO: only load the sequitur model once
# TODO: python class template and children for sequitur, fairseq, and thrax?


class Options(dict):
    """
    This class parses and passes options to other classes"
    """
    def __init__(self, modelFile="final.mdl", encoding="UTF-8",
                 variants_number=4, variants_mass=0.9):
        super(Options, self).__init__(modelFile=modelFile, encoding=encoding,
                                      variants_number=variants_number,
                                      variants_mass=variants_mass)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value


GRAMMATEK_LSTM = fs_g2p()


def pronounce(words):
    """
    Pronounce gives the IPA phonetic transcriptions of the given words.
    """
    options = Options(
        modelFile=os.getenv("G2P_MODEL", "final.mdl")
    )
    translator = Translator(SequiturTool.procureModel(options, loadG2PSample))

    for word in words:
        left = tuple(word.lower())

        output = {
            "word": word,
            "results": []
        }
        try:
            total_posterior = 0.0
            n_variants = 0
            n_best = translator.nBestInit(left)
            while (total_posterior < options.variants_mass
                   and n_variants < options.variants_number):
                try:
                    log_like, result = translator.nBestNext(n_best)
                except StopIteration:
                    break
                posterior = math.exp(log_like - n_best.logLikTotal)
                output["results"].append(
                    {"posterior": posterior, "pronunciation": " ".join(result)}
                )
                total_posterior += posterior
                n_variants += 1
        except Translator.TranslationFailure:
            pass
        yield output


def pron_to_tsv(prons):
    """
    pron_to_tsv gives the IPA phonetic transcriptions of the given words in a
    tab separated value format.
    """
    return "\n".join(
        "{w}\t{prob}\t{pron}".format(w=item["word"],
                                     prob=res["posterior"],
                                     pron=res["pronunciation"])
        for item in prons
        for res in item["results"])


@app.route("/pron/<word>", methods=["GET", "OPTIONS"])
def route_pronounce(word):
    """
    Main GET entry point - Does the important stuff
    """
    m = request.args.get("m")
    if m and m == "fairseq":
        gen_pronounce = GRAMMATEK_LSTM.pronounce
    else:
        gen_pronounce = pronounce
    # TODO: make fairseq models work with tsv
    t = request.args.get("t")
    if t and t == "tsv":
        return Response(response=pron_to_tsv(pronounce([word])),
                        status=200,
                        content_type="text/tab-separated-values")

    d = request.args.get("d")
    if d and d in GRAMMATEK_LSTM.possible_dialects:
        return jsonify(list(gen_pronounce([word], d))), 200
    return jsonify(list(gen_pronounce([word]))), 200


@app.route("/pron", methods=["POST", "OPTIONS"])
def route_pronounce_many():
    """
    Main POST entry point - Does the important stuff
    """
    content = request.get_json(force=True)
    if "words" not in content:
        return jsonify({"error": "Field 'words' missing."}), 400

    m = request.args.get("m")
    if m and m == "fairseq":
        # d = request.args.get("d")
        # if d and d in GRAMMATEK_LSTM.possible_dialects:
        gen_pronounce = GRAMMATEK_LSTM.pronounce
    else:
        gen_pronounce = pronounce
    # TODO: make fairseq models work with tsv
    t = request.args.get("t")
    if t and t == "tsv":
        return Response(response=pron_to_tsv(pronounce(content["words"])),
                        status=200,
                        content_type="text/tab-separated-values")
    d = request.args.get("d")
    if d and d in GRAMMATEK_LSTM.possible_dialects:
        return jsonify(list(gen_pronounce(content["words"], d))), 200
    return jsonify(list(gen_pronounce(content["words"]))), 200
