

set fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936
set termencoding=utf-8
set encoding=utf-8

highlight Normal ctermfg=grey ctermbg=darkblue

" vim-pathogen
execute pathogen#infect()
filetype plugin indent on
syntax on
" end vim-pathogen

au BufRead,BufNewFile *.cu set filetype=cuda

set number

" indent
set smartindent  
set tabstop=4  
set shiftwidth=4  
set expandtab  
set softtabstop=4  
" end indent

" Always show status line, even for one window
set laststatus=2
set statusline=%<%F\ %h%m%r%=%{\"[\".(&fenc==\"\"?&enc:&fenc).((exists(\"+bomb\")\ &&\ &bomb)?\",B\":\"\").\"]\ \"}%k\ %-14.(%l/%L,%c%V%)\ %P


" auto-complete(clang_complete)
let g:clang_library_path='/usr/local/Cellar/llvm/4.0.0_1/lib/libclang.dylib'
