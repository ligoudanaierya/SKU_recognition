#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<sys/stat.h>
#include<dirent.h>
#include<time.h>
#include<unistd.h>
#define PORT 23456
#define BACKLOG 1
#define MAXDATASIZE 20000
#define BUFF_SIZE 100
#define MAX_LEN 512


void copystr(char *to,char *from,int n)
{
	int i=0;
	for(;i<n;++i)
	   *to++=*from++;
	*to='\0';
}

int finddir(const char *device_id)
{
	DIR *d;
	struct dirent *dir;
	d=opendir(".");
	if(d)
	{
		while((dir=readdir(d))!=NULL)
		{
			if(strcmp(device_id,dir->d_name)==0)
				return 1;
		}
		closedir(d);
	}	
	return 0;
}

void getCurrentTimeStr(char *timestr,int len)
{
	time_t t=time(NULL);
	strftime(timestr,len-1,"%Y-%m-%d-%H-%M-%S",localtime(&t));
}
int main(int argc,char *argv[])
{
    int listenfd,connectfd;
    struct sockaddr_in server;
    struct sockaddr_in client;
    int read_count;
    int imei_len=15; // fixed length
    char buffer[MAX_LEN],device_id[imei_len+1];
    FILE *fp;
    char timestr[64]={0};
    char imgname[100];
    int rn=0;
    int cnt=0;
    int imei_ok=0;
    socklen_t addrlen;
    pid_t pid;
    memset(device_id,0,sizeof(device_id));
    if((listenfd=socket(AF_INET,SOCK_STREAM,0))==-1)
    {
        perror("creating socket failed");
        exit(1);
    }

    bzero(&server,sizeof(server));
    server.sin_family=AF_INET;
    server.sin_port=htons(PORT);
    server.sin_addr.s_addr=htonl(INADDR_ANY);
    if(bind(listenfd,(struct sockaddr*)&server,sizeof(server))==-1)
    {
        perror("binderror.");
        exit(1);
    }
    if(listen(listenfd,BACKLOG)==-1)
    {
       		perror("listen() error\n");
        	exit(1);
    }
    while(1)
    {
   	
    	addrlen=sizeof(client);
    	if((connectfd=accept(listenfd,(struct sockaddr*)&client,&addrlen))==-1)
    	{
        	perror("accept() error\n");
        	exit(1);
    	}
    	printf("get a connection from client's ip %s  port is %d\n",
            inet_ntoa(client.sin_addr),htons(client.sin_port));
	if((pid=fork())==0){
        close(listenfd);
    	while(1)
	{
        	rn=recv(connectfd,buffer,MAX_LEN-1,0);
		if(rn<0)
		{
			printf("cannot receive file\n");
			exit(1);
		}
        	buffer[rn]='\0';
        	if(rn!=0 && !imei_ok)
        	{

			copystr(device_id,buffer,imei_len);

			strcpy(imgname,device_id);

			strcat(imgname,"/");
			getCurrentTimeStr(timestr,strlen(timestr));
			strcat(imgname,timestr);
			strcat(imgname,".jpg");
			printf("time %s device_di:%s\n",timestr,device_id);	
			if(!finddir(device_id))
			{
				if(mkdir(device_id,0777)==-1)
				{
					printf("mkdir error\n");
					exit(1);
				}	

			}
			fp=fopen(imgname,"wb");
			if(fp==NULL)
			{
				printf("file open error\n");
				exit(1);
			}
            		fwrite(buffer+imei_len,1,rn-imei_len,fp);
            		bzero(buffer,sizeof(buffer));
	    		imei_ok=1;
        	}else if(rn!=0)
		{
			fwrite(buffer,1,rn,fp);
			bzero(buffer,sizeof(buffer));
		}else{
	    		++cnt;
            		printf("#%d receive over\n",cnt);
	    		imei_ok=0;
	    		memset(buffer,0,MAX_LEN);
			memset(imgname,0,100);
			memset(timestr,0,64);
	    		close(connectfd);
	    		fclose(fp);
            		exit(0);
        	}
	   } //end while
    	}//end if
	close(connectfd);
    }
    fclose(fp);
    close(connectfd); 
    close(listenfd);
    return 0;
}

