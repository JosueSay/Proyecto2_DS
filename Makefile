PYTHON ?= python3
CLEAN_SCRIPT = 01_data_cleaning/clean_data.py
SUMMARY_SCRIPT = 01_data_cleaning/data_summary.py
EDA_SCRIPT = 02_eda/eda.py
EDA_ANALYSIS_SCRIPT = 02_eda/eda_analysis.py

# colores
BLUE := \033[1;34m
YELLOW := \033[1;33m
RED := \033[1;31m
GREEN := \033[1;32m
BOLD := \033[1m
RESET := \033[0m

# target por defecto
all: clean_data summary eda eda_analysis

# Limpieza de datos
clean_data:
	@echo "$(BLUE)$(BOLD)🧹 Ejecutando limpieza de datos...$(RESET)"
	@echo "\t- Script:       $(YELLOW)$(CLEAN_SCRIPT)$(RESET)"
	@echo "\t- Contenido:    $(YELLOW)data/clean$(RESET)"
	@TMP=$$(mktemp); \
	PYTHONPATH=01_data_cleaning/core $(PYTHON) -u $(CLEAN_SCRIPT) > $$TMP 2>&1; \
	PY_EXIT_CODE=$$?; PY_OUTPUT=$$(cat $$TMP); rm -f $$TMP; \
	if [ $$PY_EXIT_CODE -ne 0 ]; then \
		echo "$(RED)❌ Error en limpieza de datos$(RESET)"; \
		echo ""; echo "$$PY_OUTPUT"; echo ""; \
		exit 1; \
	elif echo "$$PY_OUTPUT" | grep -q "CACHE_USED"; then \
		echo "$(YELLOW)ℹ️  Limpieza omitida (cache)$(RESET)"; echo ""; \
	elif echo "$$PY_OUTPUT" | grep -q "DONE"; then \
		echo "$(GREEN)✅ Limpieza completada$(RESET)"; echo ""; \
		echo "\tRevisa el contenido generado en: $(YELLOW)data/clean$(RESET)"; \
	else \
		echo "$(RED)❌ Salida inesperada del script Python$(RESET)"; \
		echo ""; echo "$$PY_OUTPUT"; echo ""; \
		exit 1; \
	fi

# Generar resumen de datos
summary:
	@echo "$(BLUE)$(BOLD)📊 Generando resumen de datos...$(RESET)"
	@echo "\t- Script:       $(YELLOW)$(SUMMARY_SCRIPT)$(RESET)"
	@echo "\t- Contenido:    $(YELLOW)data/clean$(RESET)"
	@PY_OUTPUT=$$($(PYTHON) -u $(SUMMARY_SCRIPT) 2>&1); \
	PY_EXIT_CODE=$$?; \
	if [ $$PY_EXIT_CODE -ne 0 ]; then \
		echo "$(RED)❌ Error al generar el resumen$(RESET)"; exit 1; \
	elif echo "$$PY_OUTPUT" | grep -Eq "CACHE_USED_TRAIN_VALID|CACHE_USED"; then \
		echo "$(YELLOW)ℹ️  Resumen omitido (cache)$(RESET)\n"; \
	elif echo "$$PY_OUTPUT" | grep -q "NO_TRAIN_VALID_FILES"; then \
		echo "$(RED)❌ No existen train_strat.csv/valid_strat.csv. Ejecuta primero 'make clean_data'$(RESET)"; exit 1; \
	elif echo "$$PY_OUTPUT" | grep -q "NO_CLEAN_FILE"; then \
		echo "$(RED)❌ No existe data_clean.csv. Ejecuta primero 'make clean_data'$(RESET)"; exit 1; \
	elif echo "$$PY_OUTPUT" | grep -Eq "DONE_TRAIN_VALID|DONE"; then \
		echo "$(GREEN)✅ Resumen generado correctamente$(RESET)\n"; \
		echo "\tRevisa el contenido generado en: $(YELLOW)reports/eda$(RESET)"; \
	else \
		echo "$(RED)❌ Salida inesperada del script Python$(RESET)"; exit 1; \
	fi

# EDA general
eda:
	@echo "$(BLUE)$(BOLD)📈 Ejecutando EDA 1...$(RESET)"
	@echo "\t- Script:       $(YELLOW)$(EDA_SCRIPT)$(RESET)"
	@echo "\t- Contenido:    $(YELLOW)images/eda/*.png y reports/eda$(RESET)"
	@PY_OUTPUT=$$($(PYTHON) -u $(EDA_SCRIPT) 2>&1); \
	PY_EXIT_CODE=$$?; \
	if [ $$PY_EXIT_CODE -ne 0 ]; then \
		echo "$(RED)❌ Error durante EDA$(RESET)"; exit 1; \
	elif echo "$$PY_OUTPUT" | grep -q "CACHE_USED"; then \
		echo "$(YELLOW)ℹ️  EDA omitido (cache)$(RESET)\n"; \
	elif echo "$$PY_OUTPUT" | grep -q "DONE"; then \
		echo "$(GREEN)✅ EDA completado$(RESET)\n"; \
		echo "\tRevisa el contenido generado en: $(YELLOW)images/eda/*.png y reports/eda$(RESET)"; \
	else \
		echo "$(RED)❌ Salida inesperada del script EDA$(RESET)"; exit 1; \
	fi

# Análisis EDA detallado
eda_analysis:
	@echo "$(BLUE)$(BOLD)📊 Ejecutando análisis EDA 2...$(RESET)"
	@echo "\t- Script: $(YELLOW)$(EDA_ANALYSIS_SCRIPT)$(RESET)"
	@echo "\t- Log:    $(YELLOW)reports/clean/00_analisis.log$(RESET)"
	@PY_OUTPUT=$$($(PYTHON) -u $(EDA_ANALYSIS_SCRIPT) 2>&1); \
	PY_EXIT_CODE=$$?; \
	if [ $$PY_EXIT_CODE -ne 0 ]; then \
		echo "$(RED)❌ Error durante el análisis EDA$(RESET)"; exit 1; \
	elif echo "$$PY_OUTPUT" | grep -q "CACHE_USED"; then \
		echo "$(YELLOW)ℹ️  EDA omitido, se reutilizó cache.$(RESET)\n"; \
	elif echo "$$PY_OUTPUT" | grep -q "DONE"; then \
		echo "$(GREEN)✅ EDA finalizado$(RESET)\n"; \
		echo "\tRevisa el log en: $(YELLOW)reports/clean/00_analisis.log$(RESET)"; \
	else \
		echo "$(RED)❌ Salida inesperada del script Python$(RESET)"; exit 1; \
	fi

# Limpieza total
clean:
	@echo "$(BLUE)$(BOLD)🧹 Limpiando artefactos generados...$(RESET)"
	@for dir in reports/clean reports/results cache data/clean reports/eda; do \
		if [ -d $$dir ]; then \
			echo "$(YELLOW)\t- Eliminando: $$dir$(RESET)"; \
			rm -rf $$dir; \
		else \
			echo "$(BLUE)\t- No existe: $$dir$(RESET)"; \
		fi; \
	done
	@echo "$(GREEN)✅ Limpieza completada$(RESET)"
	@echo ""

.PHONY: all clean_data summary eda eda_analysis clean
