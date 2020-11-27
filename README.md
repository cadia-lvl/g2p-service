# g2p-service

Naive Flask wrapper for
[Sequitur](https://github.com/sequitur-g2p/sequitur-g2p) and [fairseq g2p
models](https://github.com/grammatek/g2p-lstm). Exposes a simple REST API.

## Usage
Example service endpoint for Icelandic available at
https://nlp.talgreinir.is/pron (courtesy of [Tiro](https://tiro.is)) - does not
support fairseq

How do I pronounce `derp`?

    $ curl -XGET https://nlp.talgreinir.is/pron/derp | jq
    [
      {
        "results": [
          {
            "posterior": 0.9138450652404999,
            "pronunciation": "t ɛ r̥ p"
          }
        ],
        "word": "derp"
      }
    ]

Multiple word support with a POST.
    
    $ cat <<EOF | curl -XPOST -d@- https://nlp.talgreinir.is/pron | jq
    {"words": ["herp", "derp"]}
    EOF
    [
      {
        "results": [
          {
            "posterior": 0.9251423160703962,
            "pronunciation": "h ɛ r̥ p"
          }
        ],
        "word": "herp"
      },
      {
        "results": [
          {
            "posterior": 0.9138450652404999,
            "pronunciation": "t ɛ r̥ p"
          }
        ],
        "word": "derp"
      }
    ]
    
Append `?t=tsv` to get the response in the Kaldi lexicon format.

Append ?m=fairseq to use the fairseq model instead of the sequitur model

    $ cat <<EOF | curl -XPOST -d@- "http://localhost:8000/pron?m=fairseq" | jq
    {"words": ["herp", "derp"]}
    EOF
    [
      {
        "results": [
          {
            "pronunciation": "h E r_0 p"
          }
        ],
        "word": "herp"
      },
      {
        "results": [
          {
            "pronunciation": "t E r_0 p"
          }
        ],
        "word": "derp"
      }
    ]

Append ?d=north to use the northern dialect
Append ?d=north_east to use the north eastern dialect
Append ?d=south to use the southern dialect

    $ cat <<EOF | curl -XPOST -d@- "http://localhost:8000/pron?m=fairseq&d=south" | jq
    {"words": ["herp", "akureyri"]}
    EOF
    [
      {
        "results": [
          {
            "pronunciation": "h E r_0 p"
          }
        ],
        "word": "herp"
      },
      {
        "results": [
          {
            "pronunciation": "a: k Y r ei r I"
          }
        ],
        "word": "akureyri"
      }
    ]

## Steps

### Build Docker image

    docker build -t g2p-service .
    
### Run service
Train, or somehow acquire a Sequitur G2P model expose it to the container as
`/app/final.mdl`

    docker run -p 8000:8000 -v <path-to-model>:/app/final.mdl g2p-service

    docker run -p 8000:8000 -v <path-to-model>:/app/final.mdl -v <path-to-grammatek-lstm-g2p-repo>:/app/fairseq_g2p/ g2p-service


Example
    docker run -it --rm -v ${PWD}/final.mdl:/app/final.mdl -v /home/judyfong/g2p-lstm:/app/fairseq_g2p g2p-service

## LICENSE

    Copyright (C) 2019  Róbert Kjaran <robert@kjaran.com>
    Copyright (C) 2020  Judy Y Fong <lvl@judyyfong.xyz>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
