# Graph Neural Network-Based Drug–Target Interaction Prediction for Drug Repurposing in Acute Myeloid Leukaemia

## Project overview

This repository contains the implementation for an MSc Data Science dissertation investigating graph neural network-based drug–target affinity prediction for drug repurposing in acute myeloid leukaemia.

Drug molecules are represented as molecular graphs, while protein targets are initially represented using amino-acid embeddings and a one-dimensional convolutional neural network.

## Main models

The study compares:

- GCN + protein CNN
- GAT + protein CNN
- GIN + protein CNN
- Best-performing GNN + optional ESM-2 protein representation

## Datasets

The principal benchmark datasets are:

- Davis
- KIBA

The selected model will subsequently be applied to AML-related protein targets and candidate repurposing drugs.

## Evaluation

Models will be evaluated using:

- Random drug–target pair splitting
- Cold-drug splitting
- Cold-target splitting
- Mean squared error
- Root mean squared error
- Mean absolute error
- Concordance index
- Pearson correlation

## Experimental workflow

1. Prepare Davis and KIBA datasets.
2. Train the Davis GCN + CNN baseline.
3. Compare GCN, GAT and GIN drug encoders.
4. Evaluate cold-start generalisation.
5. Repeat the experiments on KIBA.
6. Optionally compare the CNN protein encoder with ESM-2.
7. Screen candidate drugs against AML-related targets.
8. Validate the highest-ranked candidates using databases and published evidence.

## Environment

The project uses the Conda environment:

```text
dti_pyg
conda activate dti_pyg
```bash
cat > results/experiment_log.csv <<'EOF'
run_id,timestamp,dataset,model,protein_encoder,split_type,seed,batch_size,epochs,learning_rate,weight_decay,best_validation_mse,test_mse,test_rmse,test_mae,concordance_index,pearson_correlation,model_path
