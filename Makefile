
all: docker/files/ft.py.enc docker/files/aws/credentials.enc

docker/files/ft.py.enc: docker/files/ft.py
	openssl enc -aes-256-cbc -in docker/files/ft.py -out docker/files/ft.py.enc -K $$mykey -iv $$iv

docker/files/aws/credentials.enc: docker/files/aws/credentials
	openssl enc -aes-256-cbc -in docker/files/aws/credentials -out docker/files/aws/credentials.enc -K $$mykey -iv $$iv

build: docker/files/ft.py.enc docker/files/aws/credentials.enc
	cd docker ; docker build -t fenyoa/ft_gpt_oss_20b_ovh_faq -f Dockerfile .

run:
	docker run --gpus all --user=42420:42420 -e iv=$$iv -e mykey=$$mykey --rm -t -i fenyoa/ft_gpt_oss_20b_ovh_faq

shell:
	docker run --gpus all --user=42420:42420 -e iv=$$iv -e mykey=$$mykey --rm -t -i fenyoa/ft_gpt_oss_20b_ovh_faq bash

push:
	echo RUN '"docker login -u fenyoa"' and enter password
	docker push fenyoa/ft_gpt_oss_20b_ovh_faq
