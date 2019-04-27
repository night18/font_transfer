def pot2png(source_path,destination_path):
	with open(source_path, 'rb') as f:
		
		byte = f.read(1)
		while byte:
			# Do stuff with byte.
			byte = f.read(1)


pot2png('olhw/1001-c.pot', '')