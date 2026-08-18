# Feature pipeline statistics

Pipeline version `featurepipeline-v1.0.0`, feature spec `featurespec-v1.1.0`.

## Coverage

| Quantity | Value |
| --- | ---: |
| Episodes selected | 13 |
| Episodes with feature records | 13 |
| Episodes transcribed | 13 |
| Transcription failures | 0 |
| Transcript segments | 743 |
| Transcribed audio (hours) | 0.8213 |

## Candidates

Counts are kept separate on purpose -- a partial-modality record is a
usable training example with a mask, but it is not a complete one.

| Status | Count |
| --- | ---: |
| Processed | 595 |
| Complete multimodal | 583 |
| Audio only | 12 |
| Text only | 0 |
| Handcrafted only (no learned modality) | 0 |
| Failed | 0 |

### By split

| Split | Count |
| --- | ---: |
| test | 268 |
| train | 260 |
| validation | 67 |

### By content type

| Content type | Count |
| --- | ---: |
| conversational | 274 |
| music | 4 |
| narrated | 314 |
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
| Hits | 0 |
| Misses | 0 |
| Hit rate | n/a |
| Total runtime (s) | 0 |

## Features with missing values

| Feature | Missing rate |
| --- | ---: |
| `sentence_end` | 1.000 |
| `context_cosine_similarity` | 0.020 |
| `semantic_change_score` | 0.020 |
| `transcript_inter_segment_pause_ms` | 0.020 |
| `chars_before` | 0.018 |
| `prior_segment_duration_ms` | 0.018 |
| `terminal_punctuation_ordinal` | 0.018 |
| `transcript_gap_before_ms` | 0.018 |
| `transcript_sentence_end` | 0.018 |
| `words_before` | 0.018 |
| `chars_after` | 0.002 |
| `following_segment_duration_ms` | 0.002 |
| `transcript_gap_after_ms` | 0.002 |
| `words_after` | 0.002 |

