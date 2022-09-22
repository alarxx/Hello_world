 // Типа БД
 const notes = [
 	{id: '1', content: 'This is a note', author: 'Adam'},
 	{id: '2', content: 'This is another note', author: 'Harlow'},
 	{id: '3', content: 'Look, another note', author: 'Riley'},
 ];

 const express = require('express');

 const app = express();
 const port = process.env.PORT || 3000;

app.get('/', (req, res)=>{
 	res.send('Hello World');
 });

/* GRAPHQL */ 
const {ApolloServer, gql} = require('apollo-server-express');
//Схема
const typeDefs = gql`
	type Note {
		id: ID!
		content: String!
		author: String!
	}
	type Mutation{
		newNote(content: String!): Note!
	}
	type Query{
		hello: String!
		notes: [Note!]!
		note(id: ID!): Note!
	}
	`;

//Функция разрешения
const resolvers = {
	Query: {
		hello: ()=>'Hello World',
		notes: ()=>notes,
		note: (parent, args) => {
			return notes.find(note => note.id === args.id);
		}
	},
	Mutation: {
		newNote: (parent, args) => {
			let noteValue = {
				id: `${notes.length + 1}`, 
				content: args.content,
				author: 'Adam'
			}
			notes.push(noteValue);
			return noteValue;
		}
	}
}
const server = new ApolloServer({typeDefs, resolvers});
server.start()
	.then(res => {
		server.applyMiddleware({app, path: '/api'});
 		app.listen({port}, ()=>console.log(res, `server is running on http://localhost:${port}${server.graphqlPath}/`));
	});


 


