
for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
    params_name=encoder_layer_${i}_ffn_fc_1.w_0
    python exchange_sent_emb.py --from_dir ./params --param_name ${params_name} --to_dir ./params_tgt
done

