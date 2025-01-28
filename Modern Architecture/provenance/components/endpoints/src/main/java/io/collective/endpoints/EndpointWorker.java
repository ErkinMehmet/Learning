package io.collective.endpoints;

import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import io.collective.articles.ArticleDataGateway;
import io.collective.restsupport.RestTemplate;
import io.collective.rss.Item;
import io.collective.workflow.Worker;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import io.collective.rss.RSS;
import io.collective.rss.Channel;
import io.collective.rss.Item;
import java.util.List;

public class EndpointWorker implements Worker<EndpointTask> {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());
    private final RestTemplate template;
    private final ArticleDataGateway gateway;

    public EndpointWorker(RestTemplate template, ArticleDataGateway gateway) {
        this.template = template;
        this.gateway = gateway;
    }

    @NotNull
    @Override
    public String getName() {
        return "ready";
    }

    @Override
    public void execute(EndpointTask task) throws IOException {
        String response = template.get(task.getEndpoint(), task.getAccept());
        gateway.clear();

        { // todo - map rss results to an article infos collection and save articles infos to the article gateway
            //get endpoint then use template to send http to get info
            try {
                String endpt=task.getEndpoint();
                String res=template.get(endpt, task.getAccept()); //Accept is just the format
                //System.out.println(res);
                // we need to parse the xml
                XmlMapper xmlMapper = new XmlMapper();
                RSS rss=xmlMapper.readValue(res,RSS.class);
                Channel ch=rss.getChannel();
                List<Item> items=ch.getItem();
                for (Item item:items) {
                    String t = item.getTitle();
                    // now we need to insert the articles
                    gateway.save(t);
                }


            } catch (Exception e) {
                logger.error(e.getMessage());
            }



        }
    }
}
