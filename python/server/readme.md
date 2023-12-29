1. Modify docker-compose.yml to change the postgres credentials (or don't)
1. Run `docker-compose up -d` to start a postgres database and the server
1. Install yandex/pgmigrate by running `pip install yandex-pgmigrate`
1. Run `$env:PGUSER='myuser'; $env:PGPASSWORD='mypassword'; pgmigrate -t 1 migrate` to create the database schema (change user and password to match the ones you set in docker-compose.yml)

Server should now be running. To test, get an X-Session-Id and an authorization token by logging in through the website. Then run the following curl command:

`curl -vvv localhost:5000/api/projects --heads 'Authorization Bearer xyz' --cookie 'X-Session-Id=xyz'`

The result will be an empty json object that would contain any projects if there were any.
