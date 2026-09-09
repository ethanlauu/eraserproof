# EraserProof

EraserProof is a local framework for experimentally and cryptographically
verifying machine unlearning.

The project studies whether an unlearned language model behaves like a model
that was never trained on the forgotten data.

## Core Models

Given:

- D_R: retained training data
- D_F: data requested for deletion

we compare:

M_full = Train(D_R ∪ D_F)

M_unlearn = Unlearn(M_full, D_F)

against the gold-standard:

M_retrain = Train(D_R)

## Goals

EraserProof will evaluate unlearning using:

- direct extraction
- token likelihood
- paraphrase attacks
- membership inference
- contextual recovery
- prefix attacks
- relearning attacks
- representation analysis
- retrain-model equivalence

Later phases will add:

- Merkle dataset commitments
- signed deletion requests
- model manifests
- cryptographic audit certificates