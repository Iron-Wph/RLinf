export PYTHONPATH=${REPO_PATH}:/opt/libero:/mnt/mnt/public/wph/codes/RoboTwin_now:$PYTHONPATH
python -c 'from robotwin.envs.vector_env import VectorEnv; print("Import success!")'