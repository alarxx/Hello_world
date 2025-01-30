#pragma once


class Data {
	private:
		int __theData;
	public:
		Data(); // Constructor
		~Data(); // Destructor
		// Setter
		void setData(const int data);

		// Getter
		int getData();

		// Also
		// call-by-referece
		// non-intuitive that we gonna change data
		// void getData(int & data);

		// better use pointer:
		void getData(int * const ptr_data); // the pointer is const
};
