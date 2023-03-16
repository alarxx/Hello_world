const nodemailer = require('nodemailer');

class MailService {
    constructor() {
        this.transporter = nodemailer.createTransport({
            service: 'gmail',
            host: process.env.SMTP_HOST,
            port: process.env.SMTP_PORT,
            secure: false,
            auth: {
                user: process.env.SMTP_USER,
                pass: process.env.SMTP_PASS
            }
        });
    }
    async sendActivationMail(to, link){
        await this.transporter.sendMail({
            from: process.env.SMTP_USER,
            to,
            subject: 'Account activation' + process.env.API_URL,
            text: 'Hello. This email is for your email verification.',
            html: `
                <div>
                    <h1>To activate it, click on the link</h1>
                    <a href="${link}">${link}</a>
                </div>>
            `
        })
    }

}

module.exports = new MailService();