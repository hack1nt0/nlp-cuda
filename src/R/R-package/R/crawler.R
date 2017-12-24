download.hentai.image <- function(init.url, output.dir='', plot.it=T) {
    library(rvest)
    library(RCurl)
    library(imager)
    system(paste('mkdir -p ', output.dir))
    i = 1
    url = init.url
    session = html_session(init.url)
    while (TRUE) {
        img.url <- session %>%
            html_node('#i3 a img') %>%
            html_attr('src')
        img.path <- paste(output.dir, '-', i, '.jpg', sep='')
        img.status <- try(img.url %>% download.file(img.path))
        if (class(img.status) == 'try-error') {
            Sys.sleep(0.6)
            cat('Retrying', url)
        } else {
            if (plot.it) {
                plot(load.image(img.path))
            }
            session <- session %>%
                follow_link(css='#i4 .sn #next')
            new.url <- session$url
            if (url == new.url) break
            url = new.url
            i = i + 1
        }
    }
}