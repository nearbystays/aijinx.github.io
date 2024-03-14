<!DOCTYPE html>
   <html>
       <body>
            <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"> </script>
            <script>
              axios.get('https://www.google.com/finance/quote/TSLA:NASDAQ')
               .then(response => {
                  const regex= /After Hours.*?\$[0-9] *\.[0-9]*/s
                  const match = response.data.match(regex)
                  if (match) {
                    const price = match[0].match(/\$[0-9] *\.[0-9]*/);
                    console.log(price[0]);
                    // iterate through and print prices with names company
                    for (let i = 0; i < price.length; i++) {
                      document.body.innerHTML += price[i];
                    }
                  }
                })
                .catch(error=> {
                    console.error(error)
                })
            </script>
        </body>
    </html>