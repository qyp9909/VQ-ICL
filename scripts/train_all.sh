cd ..
python main.py --data_name Beauty --num_intent_embeddings 512
python main.py --data_name Toys_and_Games --num_intent_embeddings 64
python main.py --data_name ml-1m --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 --num_intent_embeddings 1024

