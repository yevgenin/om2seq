FROM pytorch/pytorch AS base

RUN apt update && apt-get install -y build-essential

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

WORKDIR /app
VOLUME [ "./out" ]
RUN mkdir ./out

FROM base as dev

COPY . .

RUN chmod +x entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]