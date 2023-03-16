const LocalStrategy = require('passport-local').Strategy;

const bcrypt = require('bcrypt');

function initialize(passport, getUserByEmail, getUserById){

    async function authenticateUser(email, password, done){
        const user = getUserByEmail(email);

        console.log("Authenticate:", user, await bcrypt.compare(password, user.password))

        if(!user){
            return done('No user with that email', false);
        }

        try{
            if(await bcrypt.compare(password, user.password)){
                return done(null, user);
            } else {
                return done('Password incorrect', false)
            }
        }
        catch(e){
            console.log(e)
            return done(e)
        }
    }

    passport.use(new LocalStrategy({ usernameField: 'email' }, authenticateUser));

    // только при создании сессии, логин
    passport.serializeUser((user, done)=>{
        console.log('serializeUser:', user)
        done(null, user.id);
    })

    // Каждый раз когда клиент делает запрос
    passport.deserializeUser((id, done) => {
        console.log('deserializeUser:', id)
        return done(null, getUserById(id))
    })
}

module.exports = initialize;