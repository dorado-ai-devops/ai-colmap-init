IMAGE_NAME     := colmap-init
VERSION        := v0.1.10
REGISTRY       := localhost:5000
HELM_VALUES    := ../devops-ai-lab/manifests/helm-instant-ngp/values.yaml

.PHONY: all build tag push update-values release run

all: release

build:
	docker build --no-cache -t $(IMAGE_NAME):$(VERSION) .

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
