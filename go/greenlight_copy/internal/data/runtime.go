package data

import (
	"fmt"
	"strconv"
)

// Runtime По умолчанию Runtime это просто число
type Runtime int32

// MarshalJSON При преобразовании в JSON мы по особенному преобразуем значение
func (r Runtime) MarshalJSON() ([]byte, error) {
	jsonValue := fmt.Sprintf("%d mins", r)

	quotedJSONValue := strconv.Quote(jsonValue)

	return []byte(quotedJSONValue), nil
}
