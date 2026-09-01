# Feature pipeline statistics

Pipeline version `featurepipeline-v1.0.0`, feature spec `featurespec-v1.1.0`.

## Coverage

| Quantity | Value |
| --- | ---: |
| Episodes selected | 77 |
| Episodes with feature records | 77 |
| Episodes transcribed | 84 |
| Transcription failures | 0 |
| Transcript segments | 17056 |
| Transcribed audio (hours) | 17.7692 |

## Candidates

Counts are kept separate on purpose -- a partial-modality record is a
usable training example with a mask, but it is not a complete one.

| Status | Count |
| --- | ---: |
| Processed | 13176 |
| Complete multimodal | 12930 |
| Audio only | 246 |
| Text only | 0 |
| Handcrafted only (no learned modality) | 0 |
| Failed | 0 |

### By split

| Split | Count |
| --- | ---: |
| test | 2275 |
| train | 8892 |
| validation | 2009 |

### By content type

| Content type | Count |
| --- | ---: |
| conversational | 3240 |
| narrated | 1899 |
| podcast | 8037 |

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
| Hits | 36 |
| Misses | 272 |
| Hit rate | 0.1169 |
| Total runtime (s) | 21144.014 |

## Features with missing values

| Feature | Missing rate |
| --- | ---: |
| `sentence_end` | 1.000 |
| `context_cosine_similarity` | 0.019 |
| `semantic_change_score` | 0.019 |
| `transcript_inter_segment_pause_ms` | 0.019 |
| `chars_before` | 0.014 |
| `prior_segment_duration_ms` | 0.014 |
| `terminal_punctuation_ordinal` | 0.014 |
| `transcript_gap_before_ms` | 0.014 |
| `transcript_sentence_end` | 0.014 |
| `words_before` | 0.014 |
| `chars_after` | 0.007 |
| `following_segment_duration_ms` | 0.007 |
| `transcript_gap_after_ms` | 0.007 |
| `words_after` | 0.007 |
| `ms_until_next_speech` | 0.000 |

