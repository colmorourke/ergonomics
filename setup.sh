cd ./tf_pose/pafprocess/ && swig -python -c++ pafprocess.i && python setup.py build_ext --inplace

mkdir -p ~/.streamlit

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml