
CXX = g++  # Define the C++ compiler as g++ (adjust if needed)
BUILD_DIR = build

all: package

swig:
	swig -c++ -python swig/dSalmon.i
	mv swig/dSalmon.py swig/__init__.py

package: swig
	tar cJf dSalmon.tar.xz contrib cpp python swig README.md LICENSE setup.py

profile: main.cpp | $(BUILD_DIR)
	$(CXX) -o $(BUILD_DIR)/program main.cpp -Icpp -Icontrib/boost/include -std=c++17 -g -pg  # Enable profiling with gprof

plain: main.cpp | $(BUILD_DIR)
	$(CXX) -o $(BUILD_DIR)/program main.cpp -Icpp -Icontrib/boost/include -std=c++17 -g0

tree: tree_test.cpp | $(BUILD_DIR)
	$(CXX) -o $(BUILD_DIR)/tree tree_test.cpp -Icpp -Icontrib/boost/include -std=c++17 -g0 

clean:
	rm -rf $(BUILD_DIR)/*

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: swig profile plain tree
