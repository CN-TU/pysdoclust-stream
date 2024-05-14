
CXX = g++  # Define the C++ compiler as g++ (adjust if needed)
BUILD_DIR = build

all: package

swig:
	swig -c++ -python swig/dSalmon.i
	mv swig/dSalmon.py swig/__init__.py

package: swig
	tar cJf dSalmon.tar.xz contrib cpp python swig README.md LICENSE setup.py

clean:
	rm -rf $(BUILD_DIR)/*

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: swig