SELFDIR=$(dirname $0)
source ${SELFDIR}/all_hypparams.sh



# bash ${SELFDIR}/jvaqg_latent_scale_wo_glove.sh
bash my_eval/evaluate_ngram.sh ${DATASET} jvaqg_latent_scale_wo_glove
# bash my_eval/evaluate_consis.sh ${DATASET} jvaqg_latent_scale_wo_glove

# bash ${SELFDIR}/vaqg_s2s_latent_scale_aq_wo_glove.sh
bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_s2s_latent_scale_wo_glove aq
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_s2s_latent_scale_wo_glove aq

# bash ${SELFDIR}/vaqg_s2s_latent_scale_qa_wo_glove.sh
bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_s2s_latent_scale_wo_glove qa
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_s2s_latent_scale_wo_glove qa

# bash ${SELFDIR}/vaqg_pipe_latent_scale_aq_wo_glove.sh
bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_pipe_latent_scale_wo_glove aq 
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_pipe_latent_scale_wo_glove aq 

# bash ${SELFDIR}/vaqg_pipe_latent_scale_qa_wo_glove.sh 
bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_pipe_latent_scale_wo_glove qa 
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_pipe_latent_scale_wo_glove qa 

