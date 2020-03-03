BOOST_DIR=/usr/gapps/nexo/boost/boost_1_68_0
BOOST_LIBS=-lboost_python37 -lboost_numpy37 
PYTHON_LIBS=-lpython3.7m
PYTHON_DIR=/usr/tce/packages/python/python-3.7.2/lib/
SNIPER_DIR=/usr/gapps/nexo/sniper/install/
SNIPER_LIBS=$(SNIPER_DIR)/lib/libSniperKernel.so
ROOT_DIR=/usr/gapps/nexo/root/root-6.16.00
ROOT_LIBS=
NEXO_DIR=${HOME}/work/nexo-offline/build
NEXO_LIBS=-L$(NEXO_DIR)/lib -lEvtNavigator -lEDMUtil -lBaseEvent -lSimEvent -lPidTmvaEvent

CXX_FLAGS = -fPIC   -std=c++14 -g -ggdb

CXX_DEFINES = 

CXX_INCLUDES = -I$(NEXO_DIR)/include -I$(SNIPER_DIR)/include -I$(BOOST_DIR)/include -I$(ROOT_DIR)/include -I/usr/tce/packages/python/python-3.7.2/include/python3.7m 

all: DnnEventTagger/libDnnEventTagger.so

DnnEventTagger.o: DnnEventTagger.cc DnnEventTagger.hh
	g++ $(CXX_FLAGS) $(CXX_DEFINES) $(CXX_INCLUDES) -c -o $@ $<

DnnEventTagger/libDnnEventTagger.so: DnnEventTagger.o
	g++ -fPIC -g -ggdb  -shared -Wl,-soname,libDnnEventTagger.so -o $@ $^ -Wl,-rpath,$(NEXO_DIR)/lib:$(SNIPER_DIR)/lib64:/usr/lib64/root:$(BOOST_DIR)/lib $(NEXO_LIBS) $(SNIPER_LIBS) -L$(BOOST_DIR)/lib $(BOOST_LIBS) -L$(PYTHON_DIR) $(PYTHON_LIBS)

tar:
	tar -czf DnnEventTagger.tgz DnnEventTagger.{hh,cc} DnnEventTagger/*.py tagger-run.py Makefile
