#include "Data.h"


// Constructor
Data::Data(){}
// Destructor
Data::~Data(){}

// Setter
void Data::setData(const int data){ this->__theData = data; }
// Getter
int Data::getData(){ return this->__theData; }
void Data::getData(int * const ptr_data){ *ptr_data = this->__theData; }
