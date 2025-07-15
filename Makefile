CC=hcc            # installed by holyc-lang
SRC=mvt.HC
BIN=mvt
$(BIN): $(SRC)
	$(CC) -o $(BIN) $(SRC)
clean:
	rm -f $(BIN)
