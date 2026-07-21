#!/bin/bash

mkdir -p logs

echo "===== Début des expériences ====="

echo "===== Lancement delta 0.1 ====="
python experiments/run_llm.py \
--config configs/multi/current/lupu_2024L_exp_qwen/figure1/delta_01.yaml \
> logs/delta_01.log 2>&1

if [ $? -ne 0 ]; then
    echo "Erreur pendant delta 0.1"
    exit 1
fi


echo "===== Lancement delta 0.4 ====="
python experiments/run_llm.py \
--config configs/multi/current/lupu_2024L_exp_qwen/figure1/delta_04.yaml \
> logs/delta_04.log 2>&1

if [ $? -ne 0 ]; then
    echo "Erreur pendant delta 0.4"
    exit 1
fi


echo "===== Lancement delta 0.8 ====="
python experiments/run_llm.py \
--config configs/multi/current/lupu_2024L_exp_qwen/figure1/delta_08.yaml \
> logs/delta_08.log 2>&1

if [ $? -ne 0 ]; then
    echo "Erreur pendant delta 0.8"
    exit 1
fi

echo "===== Toutes les expériences de la figure1 sont terminées ====="

