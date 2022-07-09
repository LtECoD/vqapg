SELFDIR=$(dirname $0)
source ${SELFDIR}/all_hypparams.sh

# bash ${SELFDIR}/jvaqg_base.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} jvaqg_base
# bash my_eval/evaluate_consis.sh ${DATASET} jvaqg_base

# bash ${SELFDIR}/jvaqg_base-block-loss.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} jvaqg_base-block-loss
# bash my_eval/evaluate_consis.sh ${DATASET} jvaqg_base-block-loss

# bash ${SELFDIR}/jvaqg_latent.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} jvaqg_latent
# bash my_eval/evaluate_consis.sh ${DATASET} jvaqg_latent

bash ${SELFDIR}/jvaqg_latent_scale.sh
bash my_eval/evaluate_ngram.sh ${DATASET} jvaqg_latent_scale
bash my_eval/evaluate_consis.sh ${DATASET} jvaqg_latent_scale

# bash ${SELFDIR}/jvaqg_latent_scale-no-block-loss.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} jvaqg_latent_scale-no-block-loss
# bash my_eval/evaluate_consis.sh ${DATASET} jvaqg_latent_scale-no-block-loss

# bash ${SELFDIR}/vaqg_s2s_base_aq.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_s2s_base aq
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_s2s_base aq

# bash ${SELFDIR}/vaqg_s2s_base_qa.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_s2s_base qa
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_s2s_base qa

# bash ${SELFDIR}/vaqg_s2s_latent_aq.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_s2s_latent aq
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_s2s_latent aq

# bash ${SELFDIR}/vaqg_s2s_latent_qa.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_s2s_latent qa
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_s2s_latent qa

bash ${SELFDIR}/vaqg_s2s_latent_scale_aq.sh
bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_s2s_latent_scale aq
bash my_eval/evaluate_consis.sh ${DATASET} vaqg_s2s_latent_scale aq

# bash ${SELFDIR}/vaqg_s2s_latent_scale_qa.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_s2s_latent_scale qa
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_s2s_latent_scale qa

# bash ${SELFDIR}/vaqg_pipe_base_aq.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_pipe_base aq
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_pipe_base aq

# bash ${SELFDIR}/vaqg_pipe_base_qa.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_pipe_base qa
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_pipe_base qa

# bash ${SELFDIR}/vaqg_pipe_latent_aq.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_pipe_latent aq
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_pipe_latent aq

# bash ${SELFDIR}/vaqg_pipe_latent_qa.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_pipe_latent qa
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_pipe_latent qa

# bash ${SELFDIR}/vaqg_pipe_latent_scale_aq.sh
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_pipe_latent_scale aq 
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_pipe_latent_scale aq 

# bash ${SELFDIR}/vaqg_pipe_latent_scale_qa.sh 
# bash my_eval/evaluate_ngram.sh ${DATASET} vaqg_pipe_latent_scale qa 
# bash my_eval/evaluate_consis.sh ${DATASET} vaqg_pipe_latent_scale qa 
