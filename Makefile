
all: docker/files/ft.py.enc docker/files/aws/credentials.enc

decrypt:
	openssl enc -d -aes-256-cbc -out docker/files/ft.py -in docker/files/ft.py.enc -K $$mykey -iv $$iv

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

run-on-ovh:
	echo RUN '"ovhai login" (cf. 1Password)'
	ovhai job run --name faq --flavor h100-1-gpu --gpu 1 --ssh-public-keys "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCbmnkcOvTnqp8cBLfGhy+DAsKGlCt2xWYD0Ujv23167O5uftUDx5tdCaDoI6vdgFLgG9S0XHQNm15cmZQAfMPSDuvZvcCXC/Vy8T5jSPb7X+ly//C2VXavEq6T+OLPXw248MR14FWKuX4og2zBuzRkXJMol2+jiJYaGn7U61FjDfotjQ/04+q4niMm50tZHgRc3UJkW3p8hRrSUK1kyLK8oZzKDJslcwfNEIF6EH/FO6Uad5NBjkewCckl3ZIyKNrfGJ6uzo6hAMg4pTlWFL/niZGYlu1kc0HFW0ronaMgqHE2kgg9a6voH9MFh0iOFDCr3CI9G7K/3qd2cFYqwmPJ2MD/by8ZR0vP/fzcc4AFiF/P6kjafpAVmb1DbAllwE7EInKP6i2DzDNvs4r1gZLZ6FyXFqN/elzFwwy/zaL+MbQ436SXqbSEW4N2VbJ4W5nMo11G3gtGnlqngI1xa2QqtM1pICOSWewg1MuoduUubSAq+U0E4c8o6Q7sETaGjD8= alexandre fenyo@DESKTOP-87D5I2O" --unsecure-http fenyoa/ft_gpt_oss_20b_ovh_faq -e mykey=$$mykey -e iv=$$iv

push:
	echo RUN '"docker login -u fenyoa"' and enter password
	docker push fenyoa/ft_gpt_oss_20b_ovh_faq

