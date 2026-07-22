# Feature pipeline statistics

Pipeline version `featurepipeline-v1.0.0`, feature spec `featurespec-v1.1.0`.

## Coverage

| Quantity | Value |
| --- | ---: |
| Episodes selected | 7 |
| Episodes with feature records | 7 |
| Episodes transcribed | 7 |
| Transcription failures | 0 |
| Transcript segments | 45 |
| Transcribed audio (hours) | 0.0351 |

## Candidates

Counts are kept separate on purpose -- a partial-modality record is a
usable training example with a mask, but it is not a complete one.

| Status | Count |
| --- | ---: |
| Processed | 20 |
| Complete multimodal | 18 |
| Audio only | 2 |
| Text only | 0 |
| Handcrafted only (no learned modality) | 0 |
| Failed | 0 |

### By split

| Split | Count |
| --- | ---: |
| development | 20 |

### By content type

| Content type | Count |
| --- | ---: |
| conversational | 6 |
| music | 4 |
| narrated | 7 |
| podcast | 3 |

## Dimensions

| Quantity | Value |
| --- | --- |
| Audio embedding model | `openai/whisper-tiny.en` |
| Audio embedding dimension (raw encoder) | [384] |
| Text embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Text embedding dimension (native) | [384] |
| Constructed text vector (4 blocks) | 1536 |
| Handcrafted feature count | 110 |

The constructed text width is arithmetic, not a model property: it is
four concatenated blocks of MiniLM's native output.

## Cache

| Quantity | Value |
| --- | ---: |
| Hits | 28 |
| Misses | 0 |
| Hit rate | 1.0 |
| Total runtime (s) | 0.483 |

## Features with missing values

| Feature | Missing rate |
| --- | ---: |
| `sentence_end` | 1.000 |
| `chars_before` | 0.100 |
| `context_cosine_similarity` | 0.100 |
| `prior_segment_duration_ms` | 0.100 |
| `semantic_change_score` | 0.100 |
| `terminal_punctuation_ordinal` | 0.100 |
| `transcript_gap_before_ms` | 0.100 |
| `transcript_inter_segment_pause_ms` | 0.100 |
| `transcript_sentence_end` | 0.100 |
| `words_before` | 0.100 |

