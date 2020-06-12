mkdir -p ~/.streamlit

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

.heroku/python/bin/python -m pip install cython