RUN pip install --upgrade pip && \
    pip install \
      torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install \
      git+https://github.com/openai/CLIP.git \
      umap-learn \
      scikit-learn \
      hdbscan \
      matplotlib \
      pandas \
      tqdm \
      Pillow \
      numpy
