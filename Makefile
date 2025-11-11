.PHONY: all run profile validate docker-build docker-run clean

SUB=submissions/submission_best_pipeline.csv

all: run profile validate

run:
	@echo ">>> Build submission"
	PYTHONPATH=. python make_submission.py

profile:
	@echo ">>> Profile inference"
	PYTHONPATH=. python scripts/profile_inference.py

validate: $(SUB)
	@echo ">>> Validate submission"
	python scripts/validate_submission.py $(SUB)

docker-build:
	@echo ">>> Docker build"
	docker build -t cardi-hack:latest .

docker-run:
	@echo ">>> Docker run"
	docker run --rm \
	  -e PYTHONPATH=/app \
	  -v "$$(pwd)/data/raw:/app/data/raw" \
	  -v "$$(pwd)/submissions:/app/submissions" \
	  cardi-hack:latest

clean:
	@echo ">>> Clean submissions"
	rm -f submissions/*.csv
