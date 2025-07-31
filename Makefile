IMAGE_NAME     := colmap-init
VERSION        := v1.0.18
REGISTRY       := localhost:5000
HELM_VALUES    := ../devops-ai-lab/manifests/helm-instant-ngp/values.yaml

.PHONY: all build tag push update-values release run

all: release

build:
	docker build -t $(IMAGE_NAME):$(VERSION) .

tag: build
	docker tag $(IMAGE_NAME):$(VERSION) $(REGISTRY)/$(IMAGE_NAME):$(VERSION)

push: tag
	docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION)

update-values:
	@echo "📝 Actualizando Helm values para initContainer..."
	sed -i '/^init:/,/^[^[:space:]]/ s|^\(\s*repository:\s*\).*|\1$(REGISTRY)/$(IMAGE_NAME)|' $(HELM_VALUES)
	sed -i '/^init:/,/^[^[:space:]]/ s|^\(\s*tag:\s*\).*|\1"$(VERSION)"|' $(HELM_VALUES)


release: push update-values
	@echo "✅ InitContainer release completo: $(REGISTRY)/$(IMAGE_NAME):$(VERSION) listo para deploy."

run:
	docker run --rm \
		-v $(PWD)/data:/data \
		--gpus all \
		-e DATASET_NAME=lego-ds \
		-e DATA_PATH=/data/lego-ds \
		-e GH_KEY="xxx" \
		$(IMAGE_NAME):$(VERSION)

make-public:
	@echo "🔓 Haciendo público el paquete en GHCR..."
	curl -sSL -X PATCH https://api.github.com/user/packages/container/$(IMAGE_NAME)/visibility \
		-H "Accept: application/vnd.github+json" \
		-H "Authorization: Bearer $(GITHUB_TOKEN)" \
		-H "X-GitHub-Api-Version: 2022-11-28" \
		-d '{"visibility":"public"}' | jq '.visibility'
	@echo "✅ Paquete $(IMAGE_NAME) ahora es público en GHCR."

push-public-release: push make-public update-values
	@echo "🚀 Imagen publicada públicamente y valores de Helm actualizados."