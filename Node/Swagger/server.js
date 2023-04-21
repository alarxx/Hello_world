const express = require('express');

const swaggerUi = require('swagger-ui-express');
const swaggerDocument = require('./swagger.json');

const app = express();

app.use('/api-docs', swaggerUi.serve);
app.get('/api-docs', swaggerUi.setup(swaggerDocument));

app.get('/', (req, res)=>{
	res.json({ page: 'home' });
})

const port = process.env.PORT || 3000;
app.listen(port, ()=>{
	console.log(`server is listening on port ${port}`)
});